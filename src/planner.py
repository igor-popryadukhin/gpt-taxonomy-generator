from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Sequence

from .config import AppConfig
from .deepseek import DeepSeekClient
from .existing import ExistingCategories
from .normalize import NormalizedName, Normalizer

_logger = logging.getLogger(__name__)


@dataclass
class Node:
    path: List[str]


@dataclass
class PlanResult:
    paths: List[List[str]]
    stats: Dict[str, Any]


class CategoryPlanner:
    def __init__(
        self,
        config: AppConfig,
        client: DeepSeekClient,
        normalizer: Normalizer,
        existing: ExistingCategories,
    ) -> None:
        self.config = config
        self.client = client
        self.normalizer = normalizer
        self.existing = existing
        self._siblings: DefaultDict[str, list[NormalizedName]] = defaultdict(list)
        self._path_registry: set[str] = set()
        self.stats: Dict[str, Any] = defaultdict(int)
        self._level_counts: DefaultDict[int, int] = defaultdict(int)

    def _parent_key(self, parts: Sequence[str]) -> str:
        return self.normalizer.path_key(parts)

    def _prepare_candidate(self, raw: str, final_level: bool) -> NormalizedName | None:
        self.stats["candidates_seen"] += 1
        cleaned = self.normalizer.normalize_name(raw)
        if not cleaned:
            self.stats["filtered_empty"] += 1
            return None
        if not final_level and self.normalizer.is_stop_word(cleaned):
            self.stats["filtered_stopwords"] += 1
            return None
        slug = self.normalizer.slug(cleaned)
        canonical = self.normalizer.canonical_form(cleaned) or slug
        return NormalizedName(display=cleaned, slug=slug, canonical=canonical)

    def _register_candidate(self, parent: Sequence[str], normalized: NormalizedName) -> bool:
        parent_key = self._parent_key(parent)
        siblings = self._siblings[parent_key]
        if self.config.dedupe:
            for existing in siblings:
                if self.normalizer.are_similar(normalized, existing):
                    self.stats["filtered_duplicates"] += 1
                    return False
            full_path = parent + [normalized.display]
            full_key = self.normalizer.path_key(full_path)
            if full_key in self._path_registry:
                self.stats["filtered_duplicates"] += 1
                return False
            self._path_registry.add(full_key)
        siblings.append(normalized)
        self.stats["candidates_kept"] += 1
        return True

    async def _expand_node(self, node: Node) -> list[Node]:
        depth = len(node.path)
        final_level = depth + 1 == self.config.depth
        parent_str = " / ".join(node.path)
        _logger.info("Генерация уровня %s для '%s'", depth + 1, parent_str)
        accepted_children: list[str] = []

        # Use existing categories first
        existing_children = self.existing.children_for(node.path)
        for child in existing_children:
            if len(accepted_children) >= self.config.breadth:
                break
            prepared = self._prepare_candidate(child, final_level)
            if not prepared:
                continue
            if not self._register_candidate(node.path, prepared):
                continue
            accepted_children.append(prepared.display)
            self.stats["existing_used"] += 1

        remaining = self.config.breadth - len(accepted_children)
        ignore_list = list(dict.fromkeys(existing_children + accepted_children))
        if remaining > 0:
            request_size = max(remaining, math.ceil(self.config.breadth * 1.2))
            for attempt in range(3):
                response = await self.client.generate(
                    parent_path=parent_str,
                    count=request_size,
                    lang=self.config.lang,
                    existing=ignore_list,
                    final_level=final_level,
                    attempt=attempt,
                )
                self.stats["model_calls"] += 1
                if attempt > 0:
                    self.stats["model_retries"] += 1
                if not response.candidates:
                    self.stats["model_empty"] += 1
                for candidate in response.candidates:
                    if len(accepted_children) >= self.config.breadth:
                        break
                    prepared = self._prepare_candidate(candidate, final_level)
                    if not prepared:
                        continue
                    if not self._register_candidate(node.path, prepared):
                        continue
                    accepted_children.append(prepared.display)
                    ignore_list.append(prepared.display)
                    self.stats["model_used"] += 1
                if len(accepted_children) >= self.config.breadth:
                    break

        if len(accepted_children) < self.config.breadth:
            _logger.warning(
                "Получено %s/%s подкатегорий для '%s'", len(accepted_children), self.config.breadth, parent_str
            )

        children_nodes = [Node(path=node.path + [name]) for name in accepted_children]
        self.stats["total_children"] += len(children_nodes)
        return children_nodes

    async def build(self) -> PlanResult:
        root_name = self.normalizer.normalize_name(self.config.root)
        root_node = Node(path=[root_name])
        self._path_registry.add(self._parent_key(root_node.path))
        current_level: list[Node] = [root_node]
        self._level_counts[len(root_node.path)] = 1
        results: list[List[str]] = []

        while current_level:
            leaves = [node for node in current_level if len(node.path) == self.config.depth]
            for leaf in leaves:
                results.append(leaf.path)
            expandable = [node for node in current_level if len(node.path) < self.config.depth]
            if not expandable:
                break
            expansions = await asyncio.gather(*(self._expand_node(node) for node in expandable))
            next_level: list[Node] = []
            for children in expansions:
                if not children:
                    self.stats["dead_ends"] += 1
                for child in children:
                    next_level.append(child)
                    self._level_counts[len(child.path)] += 1
            current_level = next_level

        stats = self._compose_stats(results)
        return PlanResult(paths=results, stats=stats)

    def _compose_stats(self, paths: Sequence[Sequence[str]]) -> Dict[str, Any]:
        total_nodes = sum(self._level_counts.values())
        leaf_count = len(paths)
        internal_nodes = max(1, total_nodes - leaf_count)
        avg_branching = self.stats["total_children"] / internal_nodes if internal_nodes else 0
        return {
            "depth_target": self.config.depth,
            "depth_reached": max(self._level_counts) if self._level_counts else 0,
            "nodes_total": total_nodes,
            "leaf_nodes": leaf_count,
            "avg_branching": round(avg_branching, 2),
            "candidates_seen": self.stats.get("candidates_seen", 0),
            "candidates_kept": self.stats.get("candidates_kept", 0),
            "filtered_empty": self.stats.get("filtered_empty", 0),
            "filtered_stopwords": self.stats.get("filtered_stopwords", 0),
            "filtered_duplicates": self.stats.get("filtered_duplicates", 0),
            "existing_used": self.stats.get("existing_used", 0),
            "model_used": self.stats.get("model_used", 0),
            "model_calls": self.stats.get("model_calls", 0),
            "model_retries": self.stats.get("model_retries", 0),
            "dead_ends": self.stats.get("dead_ends", 0),
        }


__all__ = ["CategoryPlanner", "PlanResult"]
