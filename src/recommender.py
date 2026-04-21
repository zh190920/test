import pandas as pd
from .user_profile import UserProfile
from typing import List, Dict

class CourseRecommender:
    def __init__(self, courses_df: pd.DataFrame, user_profile: UserProfile):
        self.courses = courses_df
        self.user_profile = user_profile

    def recommend(self, user_profile: Dict[str, List[str]], top_k: int = 5) -> List[Dict]:
        recommendations = []
        for _, course in self.courses.iterrows():
            score = self._calculate_score(user_profile, course)
            if score > 0:
                recommendations.append({
                    'title': course['课程'],
                    'score': score
                })
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]

    def _calculate_score(self, profile: Dict[str, List[str]], course) -> float:
        score = 0
        course_tags = course['标签'].split(',')
        for category, tags in profile.items():
            if not tags:
                continue
            if not isinstance(tags, list):
                continue
            for tag in tags:
                if tag in course_tags:
                    score += 1  # 简单匹配得分
        return score