import pandas as pd
from .tag_tree import TagTree
from typing import Dict, List
import json

class UserProfile:
    def __init__(self, tag_tree: TagTree):
        self.tag_tree = tag_tree

    def build_profile(self, user_data: Dict) -> Dict[str, List[str]]:
        profile = {}
        # 提取标签
        tags = self._extract_tags(user_data)
        import datetime
        LEVEL1_TAGS = [
            '身份标签', '岗位职能', '核心产品', '核心技术', '技能层级', '学习成长'
        ]
        tree = self.tag_tree.tree
        # 1. LLM结构化推理
        tree_str = str(list(tree.edges()))
        try:
            llm_output = self.tag_tree.reason_with_llm(str(user_data), tree_str)
            llm_profile = self.extract_dict_from_llm_output(llm_output)
        except Exception as e:
            print(f"[ERROR] LLM output extraction: {e}")
            llm_profile = {}

            # 2. 分层对比与相似性判断
            for level_tag in LEVEL1_TAGS:
                user_value = None
                # 优先用 LLM 推理结果
                if level_tag in llm_profile and llm_profile[level_tag]:
                    user_value = llm_profile[level_tag]
                # 兼容字符串/列表
                if isinstance(user_value, str):
                    user_value = [user_value]
                if not user_value:
                    profile[level_tag] = None
                    continue
                # 与标签树该层所有节点对比
                candidates = [n for n in tree.successors(level_tag)]
                matched = []
                for uv in user_value:
                    found = False
                    for cand in candidates:
                        # 计算embedding和关键词相似度
                        sim = 0.7 * self._embedding_sim(uv, cand) + 0.3 * self._keyword_sim(uv, cand)
                        if sim > 0.9:
                            matched.append(cand)
                            found = True
                            break
                    if not found:
                        matched.append(uv)  # 视为新标签
                profile[level_tag] = matched if matched else None

            # 3. 最大路径匹配与标签树扩展
            new_nodes = []
            for level_tag in LEVEL1_TAGS:
                if profile[level_tag]:
                    for tag in profile[level_tag]:
                        if tag not in tree:
                            # 新增节点
                            self.tag_tree.add_tag(level_tag, tag)
                            new_nodes.append((level_tag, tag))

            # 4. 若有新增节点，保存新标签树到md
            if new_nodes:
                today = datetime.datetime.now().strftime('%Y-%m-%d')
                md_path = f'src/tree/{today}_user_tag_tree.md'
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(self._tree_to_md(tree))

            # 5. 保证输出为6级标签结构，没有的用None
            result = {}
            for tag in LEVEL1_TAGS:
                if tag in profile and profile[tag]:
                    result[tag] = profile[tag]
                else:
                    result[tag] = None
            return result

        def _embedding_sim(self, t1, t2):
            try:
                v1 = self.tag_tree._get_embedding(t1)
                v2 = self.tag_tree._get_embedding(t2)
                v1 = np.array(v1)
                v2 = np.array(v2)
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            except Exception as e:
                print(f"[WARN] embedding sim error: {e}")
                return 0.0

        def _keyword_sim(self, t1, t2):
            kw1 = set(str(t1).split())
            kw2 = set(str(t2).split())
            if not kw1 or not kw2:
                return 0.0
            return len(kw1 & kw2) / len(kw1 | kw2)

        def _tree_to_md(self, tree):
            # 递归输出树为md格式
            def dfs(node, depth):
                lines = []
                for child in tree.successors(node):
                    lines.append('  ' * depth + f'- {child}')
                    lines.extend(dfs(child, depth + 1))
                return lines
            lines = ['- root']
            lines.extend(dfs('root', 1))
            return '\n'.join(lines)
        # 匹配路径
        matched = self.tag_tree.match_path(tags)
        profile.update(matched)
        # 融合精确标签
        for category, tag_list in profile.items():
            profile[category] = self.tag_tree.fuse_precise_tags(tag_list)
        return profile

    def _extract_tags(self, user_data: Dict) -> List[str]:
        tags = []
        # 简单提取逻辑，实际可使用NLP
        if 'behavior_tags' in user_data:
            tags.extend(user_data['behavior_tags'].split(','))
        if 'job_level' in user_data:
            tags.append(user_data['job_level'])
        if 'tech_domain' in user_data:
            tags.append(user_data['tech_domain'])
        return tags