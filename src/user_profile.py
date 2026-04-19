import pandas as pd
from .tag_tree import TagTree
from typing import Dict, List
import json

class UserProfile:
    def __init__(self, tag_tree: TagTree):
        self.tag_tree = tag_tree

    def build_profile(self, user_data: Dict) -> Dict[str, List[str] or None]:
        # 6级标签名
        LEVEL1_TAGS = [
            '身份标签', '岗位职能', '核心产品', '核心技术', '技能层级', '学习成长'
        ]
        profile = {}
        # 提取标签
        tags = self._extract_tags(user_data)
        # 使用LLM推理结构（随机样本）
        tree_str = str(list(self.tag_tree.tree.edges()))
        llm_output = self.tag_tree.reason_with_llm(str(user_data), tree_str)
        try:
            llm_profile = json.loads(llm_output)
            # 递归匹配和更新标签树
            self.tag_tree.match_and_update_path(llm_profile)
        except:
            pass
        # 匹配路径
        matched = self.tag_tree.match_path(tags)
        profile.update(matched)
        # 融合精确标签
        for category, tag_list in profile.items():
            profile[category] = self.tag_tree.fuse_precise_tags(tag_list)

        # 保证输出为6级标签结构，没有的用None
        result = {}
        for tag in LEVEL1_TAGS:
            if tag in profile and profile[tag]:
                result[tag] = profile[tag]
            else:
                result[tag] = None
        return result

    def _extract_tags(self, user_data: Dict) -> List[str]:
        tags = []
        # 优先处理自然语言描述
        if 'description' in user_data:
            # 这里直接用LLM推理，或简单分词（可扩展为NLP/实体识别）
            desc = user_data['description']
            # 简单分割，实际可用NLP
            tags = [w.strip() for w in desc.replace('，', ',').replace('。', ',').replace(' ', ',').split(',') if w.strip()]
            return tags
        # 兼容老字段
        if 'behavior_tags' in user_data:
            tags.extend(user_data['behavior_tags'].split(','))
        if 'job_level' in user_data:
            tags.append(user_data['job_level'])
        if 'tech_domain' in user_data:
            tags.append(user_data['tech_domain'])
        return tags