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
        self.tree = nx.DiGraph()
        self.vectors: Dict[str, List[float]] = {}
        self.keywords: Dict[str, List[str]] = {}  # 关键词字典
        self._init_tree()
        self.llm = OpenAI(api_key="sk-wgholxssmljlxivhonnjivryzmoxzfbfxunpswfmncuaydlx", base_url="https://api.siliconflow.cn/v1")

    def match_structured_tags(self, user_input: str) -> Dict[str, str]:
            """
            先用LLM将用户输入结构化为各层级标签，然后对每个层级做相似度匹配。
            返回：{层级: 匹配到的标签}
            """
            json_path = os.path.join(os.path.dirname(__file__), 'tree', 'user_tag_tree.json')
            md_path = os.path.join(os.path.dirname(__file__), 'tree', 'user_tag_tree.md')
            with open(json_path, 'r', encoding='utf-8') as f:
                tree_json = json.load(f)
            # 1. 用LLM结构化
            structured = self._extract_structured_tags(user_input, tree_json)
            matched = {}
            for layer, tag in structured.items():
                candidates = tree_json.get(layer, [])
                sims = [(c, self._similarity(tag, c)) for c in candidates]
                sims.sort(key=lambda x: x[1], reverse=True)
                if sims and sims[0][1] > 0.75:
                    matched[layer] = sims[0][0]
                else:
                    self.add_tag(layer, tag)
                    self._update_md(md_path, layer, tag)
                    matched[layer] = tag
            return matched

    def _extract_structured_tags(self, user_input: str, tree_json: dict) -> Dict[str, str]:
        """
        用LLM将用户输入结构化为各层级标签，若LLM不可用则用简单规则。
        """
        # 优先用reason_with_llm
        try:
            tree_md_path = os.path.join(os.path.dirname(__file__), 'tree', 'user_tag_tree.md')
            llm_result = self.reason_with_llm(user_input, tree_md_path)
            # 解析LLM输出为dict
            import ast
            if isinstance(llm_result, str):
                llm_result = llm_result.strip()
                if llm_result.startswith('{'):
                    structured = ast.literal_eval(llm_result)
                else:
                    structured = {}
            else:
                structured = llm_result
            # 展平成一层dict
            flat = {}
            def _flatten(d, prefix=None):
                for k, v in d.items():
                    if isinstance(v, dict):
                        _flatten(v, k)
                    else:
                        flat[prefix or k] = v
            _flatten(structured)
            # 只保留tree_json中有的层级
            return {k: v for k, v in flat.items() if k in tree_json and v not in (None, '', 'None')}
        except Exception:
            # fallback: 简单规则
            result = {}
            for layer in tree_json:
                for tag in tree_json[layer]:
                    if tag in user_input:
                        result[layer] = tag
            return result

    def _init_tree(self):
        # 初始化标签树结构
        self.tree.add_node('root')
        self.tree.add_edge('root', '身份岗位标签')
        self.tree.add_edge('root', '岗位职能标签')
        self.tree.add_edge('root', '核心产品标签')
        self.tree.add_edge('root', '核心技术标签')
        self.tree.add_edge('root', '技能层级标签')
        self.tree.add_edge('root', '学习成长标签')

        # 示例节点
        self.tree.add_edge('身份岗位标签', '初级工程师')
        self.tree.add_edge('岗位职能标签', '伺服调试')
        self.tree.add_edge('核心产品标签', '汇川伺服通用系列')
        self.tree.add_edge('核心技术标签', '故障排查')
        self.tree.add_edge('技能层级标签', '入门')
        self.tree.add_edge('学习成长标签', '碎片化学习')

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
        """
        逐层比对user_tag_tree.json，最大匹配路径下新建节点，并同步更新user_tag_tree.md。
        """
        matched = {}
        json_path = os.path.join(os.path.dirname(__file__), 'tree', 'user_tag_tree.json')
        md_path = os.path.join(os.path.dirname(__file__), 'tree', 'user_tag_tree.md')
        with open(json_path, 'r', encoding='utf-8') as f:
            tree_json = json.load(f)
        # 层级顺序
        layers = list(tree_json.keys())
        for tag in user_tags:
            path = ['root']
            parent = 'root'
            found = False
            for layer in layers:
                candidates = tree_json[layer]
                sims = [(c, self._similarity(tag, c)) for c in candidates]
                sims.sort(key=lambda x: x[1], reverse=True)
                if sims and sims[0][1] > 0.75:
                    path.append(sims[0][0])
                    parent = sims[0][0]
                    found = True
                else:
                    # 在parent下新建tag
                    self.add_tag(parent, tag)
                    path.append(tag)
                    self._update_md(md_path, parent, tag)
                    found = False
                    break
            if found:
                # 路径完全匹配
                category = path[1] if len(path) > 1 else '其他'
                matched.setdefault(category, []).append(tag)
            else:
                # 新增节点
                category = path[1] if len(path) > 1 else '其他'
                matched.setdefault(category, []).append(tag)
        return matched

    def _similarity(self, tag1: str, tag2: str) -> float:
        # 计算两个标签的相似度（向量+关键词）
        vec1 = self._get_embedding(tag1)
        vec2 = self._get_embedding(tag2)
        vec_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        kw_sim = self._keyword_similarity(tag1.split(), tag2.split())
        return 0.7 * vec_sim + 0.3 * kw_sim

    def _update_md(self, md_path: str, parent: str, tag: str):
        # 在md文档中parent下添加tag节点
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        insert_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(parent + ' -->'):
                insert_idx = i
        if insert_idx != -1:
            # 插入新节点
            lines.insert(insert_idx + 1, f'    {parent} --> {tag}\n')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

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

    def reason_with_llm(self, user_input: str, tree_md_path: str) -> str:
        # 读取完整标签树结构
        with open(r"E:\vs_git\test\src\tree\user_tag_tree.md", 'r', encoding='utf-8') as f:
            full_structure = f.read()
        output_format = {"身份岗位标签": {"初级工程师": {"岗位职能标签": {"伺服调试工程师": {"核心产品标签": {"汇川伺服系列": {"核心技术标签": {"故障排查": {"技能层级标签": {"进阶":  {"学习成长标签": {"学习目的": "解决工作难题", "学习习惯": "录播实操训练营", "付费意愿": "中" }}}}}}}}}}}}
        output_format2 = {"身份岗位标签": {"中级工程师": {"岗位职能标签": {"伺服调试工程师": {"核心产品标签": {"汇川伺服系列": {"核心技术标签": {"参数调试": {"技能层级标签": {"None":  {"学习成长标签": {"学习目的": "None", "学习习惯": "None", "付费意愿": "None" }}}}}}}}}}}}
        output_format3 = {"身份岗位标签": {"中级工程师": {"岗位职能标签": {"伺服调试工程师": {"核心产品标签": {"汇川伺服系列": {"核心技术标签": {"None": {"技能层级标签": {"None":  {"学习成长标签": {"学习目的": "None", "学习习惯": "None", "付费意愿": "None" }}}}}}}}}}},
                          "技术负责人": {"岗位职能标签": {"变频器工程师": {"核心产品标签": {"汇川变频器系列": {"核心技术标签": {"故障排查": {"技能层级标签": {"None":  {"学习成长标签": {"学习目的": "None", "学习习惯": "None", "付费意愿": "None" }}}}}}}}}}}
        prompt = f"""
        ## 基于用户的信息，生成该用户画像的标签。
        ## 示例1：
            输入：初级工程师，主要负责伺服调试工作，常用的核心产品是汇川伺服系列，核心技术是故障排查，技能层级处于进阶阶段，学习成长方面的学习目的是解决工作难题，学习习惯是录播实操训练营，付费意愿为中等。
            输出：{str(output_format)}

        ## 示例2：
            输入：该用户是中级工程师，一个月浏览了10次汇川伺服板块，经常搜索伺服调试相关内容，最近参加了伺服通信模块Ethercat的线上课程
            输出：{str(output_format2)}

        ## 示例3：
            输入：该用户是中级工程师，同时也是技术负责人，一个月浏览了10次汇川伺服板块和5次变频器md510相关内容，经常搜索伺服调试相关内容，在变频器故障排查板块下面发表评论2次
            输出：{str(output_format3)} 

        ## 注意
            1. 用户没有的标签可以输出None
            2. 用户可能某个节点有多个标签
            3. 输出格式：JSON格式的标签字典。

        ## 用户输入：{user_input}

        """
        response = self.llm.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message

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