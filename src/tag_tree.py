import networkx as nx
import openai
import numpy as np
from typing import Dict, List, Any, Tuple
import os
import random
import json
from openai import OpenAI

class TagTree:
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key
        self.tree = nx.DiGraph()
        self.vectors: Dict[str, List[float]] = {}
        self.keywords: Dict[str, List[str]] = {}  # 关键词字典
        self._init_tree()
        self.llm = OpenAI(api_key="openai_api_key", base_url="https://api.siliconflow.cn/v1")

    def _init_tree(self):
        # 初始化标签树结构
        self.tree.add_node('root')
        self.tree.add_edge('root', '身份标签')
        self.tree.add_edge('root', '岗位职能')
        self.tree.add_edge('root', '核心产品')
        self.tree.add_edge('root', '核心技术')
        self.tree.add_edge('root', '技能层级')
        self.tree.add_edge('root', '学习成长')

        # 示例节点
        self.tree.add_edge('身份标签', '初级工程师')
        self.tree.add_edge('岗位职能', '伺服调试')
        self.tree.add_edge('核心产品', '汇川伺服通用系列')
        self.tree.add_edge('核心技术', '故障排查')
        self.tree.add_edge('技能层级', '入门')
        self.tree.add_edge('学习成长', '碎片化学习')

        # 编码向量和关键词
        for node in self.tree.nodes():
            if node != 'root':
                self.vectors[node] = self._get_embedding(node)
                self.keywords[node] = node.split()  # 简单关键词

    def _get_embedding(self, text: str) -> List[float]:
        embedding_model = OpenAI(api_key="sk-wgholxssmljlxivhonnjivryzmoxzfbfxunpswfmncuaydlx", 
                                 base_url="https://api.siliconflow.cn/v1/")
        response =embedding_model.embeddings.create(input=text, model="BAAI/bge-m3")

        return response.data[0].embedding

    def add_tag(self, parent: str, tag: str):
        if tag not in self.tree.nodes():
            self.tree.add_edge(parent, tag)
            self.vectors[tag] = self._get_embedding(tag)
            self.keywords[tag] = tag.split()

    def find_similar(self, tag: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tag_vec = self._get_embedding(tag)
        tag_keywords = tag.split()
        similarities = []
        for node, vec in self.vectors.items():
            vec_sim = np.dot(tag_vec, vec) / (np.linalg.norm(tag_vec) * np.linalg.norm(vec))
            kw_sim = self._keyword_similarity(tag_keywords, self.keywords[node])
            combined_sim = 0.7 * vec_sim + 0.3 * kw_sim  # 加权得分
            similarities.append((node, combined_sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _keyword_similarity(self, kw1: List[str], kw2: List[str]) -> float:
        intersection = set(kw1) & set(kw2)
        union = set(kw1) | set(kw2)
        return len(intersection) / len(union) if union else 0

    def get_path(self, tag: str) -> List[str]:
        if tag in self.tree:
            return nx.shortest_path(self.tree, 'root', tag)
        return []

    def match_path(self, user_tags: List[str]) -> Dict[str, List[str]]:
        matched = {}
        for tag in user_tags:
            # 完全匹配路径
            exact_path = self._find_exact_path(tag)
            if exact_path:
                category = exact_path[1]
                matched.setdefault(category, []).append(tag)
            else:
                # 最大匹配路径
                max_path = self._find_max_path(tag)
                if max_path:
                    category = max_path[1]
                    matched.setdefault(category, []).append(tag)
                    # 在路径下新增节点
                    self.add_tag(max_path[-1], tag)
                else:
                    # 新增根节点
                    self.add_tag('root', tag)
                    matched.setdefault('其他', []).append(tag)
        return matched

    def _find_exact_path(self, tag: str) -> List[str]:
        # 强关键词匹配
        for node in self.tree.nodes():
            if node != 'root' and tag in node:
                return self.get_path(node)
        return []

    def _find_max_path(self, tag: str) -> List[str]:
        similar = self.find_similar(tag, top_k=1)
        if similar and similar[0][1] > 0.5:
            return self.get_path(similar[0][0])
        return []

    def fuse_precise_tags(self, tags: List[str]) -> List[str]:
        # 合并相似标签
        fused = []
        for tag in tags:
            if not any(self._is_similar(tag, f) for f in fused):
                fused.append(tag)
        return fused

    def _is_similar(self, t1: str, t2: str) -> bool:
        sim = self.find_similar(t1, top_k=1)
        return sim and sim[0][1] > 0.8 and sim[0][0] == t2

    def reason_with_llm(self, user_input: str, tree_structure: str) -> str:
        # 随机选择部分结构进行推理
        sample_edges = random.sample(list(self.tree.edges()), min(5, len(self.tree.edges())))
        sample_structure = str(sample_edges)
        prompt = f"""
        基于以下标签树结构样本，为用户输入生成一致的标签结构。
        标签树样本：
        {sample_structure}
        用户输入：{user_input}
        输出格式：JSON格式的标签字典。
        """
        response = self.llm.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    def update_tree(self, updates: Dict[str, Any]):
        # 批量增删改查
        for action, data in updates.items():
            if action == 'add':
                self.add_tag(data['parent'], data['tag'])
            elif action == 'delete':
                if data in self.tree.nodes():
                    self.tree.remove_node(data)
                    del self.vectors[data]
                    del self.keywords[data]
            elif action == 'update':
                old_tag = data['old']
                new_tag = data['new']
                if old_tag in self.tree.nodes():
                    # 重命名节点
                    self.tree = nx.relabel_nodes(self.tree, {old_tag: new_tag})
                    self.vectors[new_tag] = self._get_embedding(new_tag)
                    self.keywords[new_tag] = new_tag.split()
                    del self.vectors[old_tag]
                    del self.keywords[old_tag]