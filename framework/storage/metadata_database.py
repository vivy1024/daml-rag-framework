# -*- coding: utf-8 -*-
"""
Metadata Database - 元数据数据库（SQLite）

设计原则：
- 轻量级：使用SQLite，无需额外服务
- 事务性：支持ACID事务
- 高性能：索引优化，查询<10ms

作者：BUILD_BODY Team
版本：v1.0.0
日期：2025-10-28
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class MetadataDB:
    """
    元数据数据库管理器（通用框架）
    
    功能：
    1. 用户统计：total_interactions, avg_reward, reputation_score
    2. 工具链统计：tools_chain, alpha, beta（Thompson Sampling参数）
    3. 模型性能：model_name, total_count, avg_reward, recent_rewards
    4. 缓存管理：cache_key, result, expires_at（TTL缓存）
    
    设计原则：
    - 零领域依赖：不硬编码健身、教育等领域知识
    - 并发安全：支持多线程读写
    - 自动迁移：版本管理，自动执行SQL迁移
    """
    
    def __init__(self, db_path: str = "data/metadata.db"):
        """
        初始化元数据数据库
        
        Args:
            db_path: 数据库文件路径（相对或绝对）
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"✅ MetadataDB initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """
        获取数据库连接（上下文管理器）
        
        使用：
            with self._get_connection() as conn:
                conn.execute(...)
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 支持字典式访问
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库Schema"""
        with self._get_connection() as conn:
            # 用户统计表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    total_interactions INTEGER DEFAULT 0,
                    avg_reward REAL,
                    teacher_usage_count INTEGER DEFAULT 0,
                    student_usage_count INTEGER DEFAULT 0,
                    last_interaction_at INTEGER,
                    reputation_score REAL DEFAULT 1.0,
                    created_at INTEGER,
                    updated_at INTEGER
                )
            """)
            
            # 工具链统计表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_chain_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    tools_chain TEXT,
                    alpha INTEGER DEFAULT 1,
                    beta INTEGER DEFAULT 1,
                    total_count INTEGER DEFAULT 0,
                    avg_reward REAL,
                    last_used_at INTEGER,
                    created_at INTEGER,
                    UNIQUE(user_id, tools_chain)
                )
            """)
            
            # 模型性能表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    model_name TEXT,
                    total_count INTEGER DEFAULT 0,
                    avg_reward REAL,
                    recent_rewards TEXT,
                    last_used_at INTEGER,
                    created_at INTEGER,
                    UNIQUE(user_id, model_name)
                )
            """)
            
            # 缓存表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT,
                    params_hash TEXT,
                    result TEXT,
                    created_at INTEGER,
                    expires_at INTEGER
                )
            """)
            
            # 用户元学习统计表（v2.0新增）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_meta_learning_stats (
                    user_id TEXT PRIMARY KEY,
                    high_quality_samples INTEGER DEFAULT 0,
                    total_samples INTEGER DEFAULT 0,
                    current_phase TEXT DEFAULT 'teaching',
                    phase_start_time REAL,
                    last_update_time REAL,
                    student_model_success_rate REAL DEFAULT 0.0,
                    teacher_model_success_rate REAL DEFAULT 0.0
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_stats_user_id 
                ON user_stats(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_chain_user 
                ON tool_chain_stats(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_perf_user 
                ON model_performance(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON mcp_cache(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_meta_learning_phase 
                ON user_meta_learning_stats(current_phase)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_meta_learning_samples 
                ON user_meta_learning_stats(high_quality_samples)
            """)
            
            logger.info("✅ Database schema initialized")
    
    # ============================================================
    # 用户统计方法
    # ============================================================
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """
        获取用户统计信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            Dict: 统计信息，不存在返回None
                {
                    "user_id": "zhangsan",
                    "total_interactions": 100,
                    "avg_reward": 4.2,
                    "teacher_usage_count": 10,
                    "student_usage_count": 90,
                    "last_interaction_at": 1698765432,
                    "reputation_score": 1.2,
                    "created_at": 1698765432,
                    "updated_at": 1698765432
                }
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_stats WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def upsert_user_stats(
        self,
        user_id: str,
        **kwargs
    ):
        """
        插入/更新用户统计
        
        Args:
            user_id: 用户ID
            **kwargs: 统计字段
                total_interactions: int
                avg_reward: float
                teacher_usage_count: int
                student_usage_count: int
                last_interaction_at: int
                reputation_score: float
        
        Example:
            db.upsert_user_stats(
                user_id="zhangsan",
                total_interactions=100,
                avg_reward=4.2
            )
        """
        with self._get_connection() as conn:
            # 检查是否存在
            existing = self.get_user_stats(user_id)
            
            if existing:
                # 更新
                set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
                set_clause += ", updated_at = ?"
                values = list(kwargs.values()) + [int(time.time()), user_id]
                
                conn.execute(
                    f"UPDATE user_stats SET {set_clause} WHERE user_id = ?",
                    values
                )
            else:
                # 插入
                kwargs.update({
                    "user_id": user_id,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time())
                })
                
                columns = ", ".join(kwargs.keys())
                placeholders = ", ".join("?" * len(kwargs))
                
                conn.execute(
                    f"INSERT INTO user_stats ({columns}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
    
    def increment_user_interaction(
        self,
        user_id: str,
        model_used: str,
        reward: Optional[float] = None
    ):
        """
        增加用户交互计数
        
        Args:
            user_id: 用户ID
            model_used: 使用的模型（"teacher"或"student"）
            reward: 奖励值（可选）
        """
        stats = self.get_user_stats(user_id) or {
            "total_interactions": 0,
            "teacher_usage_count": 0,
            "student_usage_count": 0,
            "avg_reward": 0.0
        }
        
        # 更新计数
        stats["total_interactions"] += 1
        
        if model_used == "teacher":
            stats["teacher_usage_count"] = stats.get("teacher_usage_count", 0) + 1
        elif model_used == "student":
            stats["student_usage_count"] = stats.get("student_usage_count", 0) + 1
        
        # 更新平均奖励
        if reward is not None:
            old_avg = stats.get("avg_reward", 0.0)
            n = stats["total_interactions"]
            stats["avg_reward"] = (old_avg * (n - 1) + reward) / n
        
        stats["last_interaction_at"] = int(time.time())
        
        self.upsert_user_stats(user_id, **stats)
    
    # ============================================================
    # 工具链统计方法
    # ============================================================
    
    def get_tool_chain_stats(
        self,
        user_id: str,
        tools_chain: List[str]
    ) -> Optional[Dict]:
        """
        获取工具链统计
        
        Args:
            user_id: 用户ID
            tools_chain: 工具链（会自动排序）
        
        Returns:
            Dict: 统计信息
                {
                    "user_id": "zhangsan",
                    "tools_chain": "[\"tool_a\", \"tool_b\"]",
                    "alpha": 8,
                    "beta": 3,
                    "total_count": 10,
                    "avg_reward": 4.2,
                    "last_used_at": 1698765432
                }
        """
        tools_chain_str = json.dumps(sorted(tools_chain))
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tool_chain_stats WHERE user_id = ? AND tools_chain = ?",
                (user_id, tools_chain_str)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def upsert_tool_chain_stats(
        self,
        user_id: str,
        tools_chain: List[str],
        **kwargs
    ):
        """
        插入/更新工具链统计
        
        Args:
            user_id: 用户ID
            tools_chain: 工具链
            **kwargs: 统计字段（alpha, beta, total_count, avg_reward）
        """
        tools_chain_str = json.dumps(sorted(tools_chain))
        
        with self._get_connection() as conn:
            existing = self.get_tool_chain_stats(user_id, tools_chain)
            
            if existing:
                # 更新
                set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
                set_clause += ", last_used_at = ?"
                values = list(kwargs.values()) + [int(time.time()), user_id, tools_chain_str]
                
                conn.execute(
                    f"UPDATE tool_chain_stats SET {set_clause} WHERE user_id = ? AND tools_chain = ?",
                    values
                )
            else:
                # 插入
                kwargs.update({
                    "user_id": user_id,
                    "tools_chain": tools_chain_str,
                    "created_at": int(time.time()),
                    "last_used_at": int(time.time())
                })
                
                columns = ", ".join(kwargs.keys())
                placeholders = ", ".join("?" * len(kwargs))
                
                conn.execute(
                    f"INSERT INTO tool_chain_stats ({columns}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
    
    def update_tool_chain_beta(
        self,
        user_id: str,
        tools_chain: List[str],
        success: bool,
        reward: Optional[float] = None
    ):
        """
        更新工具链Beta分布参数（Thompson Sampling）
        
        Args:
            user_id: 用户ID
            tools_chain: 工具链
            success: 是否成功（reward >= 4.0）
            reward: 奖励值（用于计算avg_reward）
        """
        stats = self.get_tool_chain_stats(user_id, tools_chain) or {
            "alpha": 1,
            "beta": 1,
            "total_count": 0,
            "avg_reward": 0.0
        }
        
        # 更新Beta参数
        if success:
            stats["alpha"] += 1
        else:
            stats["beta"] += 1
        
        stats["total_count"] += 1
        
        # 更新平均奖励
        if reward is not None:
            old_avg = stats.get("avg_reward", 0.0)
            n = stats["total_count"]
            stats["avg_reward"] = (old_avg * (n - 1) + reward) / n
        
        self.upsert_tool_chain_stats(user_id, tools_chain, **stats)
    
    def get_all_tool_chains(self, user_id: str) -> List[Dict]:
        """
        获取用户所有工具链统计
        
        Args:
            user_id: 用户ID
        
        Returns:
            List[Dict]: 工具链列表
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tool_chain_stats WHERE user_id = ? ORDER BY total_count DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # 模型性能方法
    # ============================================================
    
    def get_model_performance(
        self,
        user_id: str,
        model_name: str
    ) -> Optional[Dict]:
        """
        获取模型性能统计
        
        Args:
            user_id: 用户ID
            model_name: 模型名称
        
        Returns:
            Dict: 性能统计
                {
                    "user_id": "zhangsan",
                    "model_name": "ollama",
                    "total_count": 90,
                    "avg_reward": 4.2,
                    "recent_rewards": "[4.5, 4.2, 4.0, ...]",
                    "last_used_at": 1698765432
                }
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM model_performance WHERE user_id = ? AND model_name = ?",
                (user_id, model_name)
            )
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # 解析JSON
                if result.get("recent_rewards"):
                    result["recent_rewards"] = json.loads(result["recent_rewards"])
                return result
            return None
    
    def update_model_performance(
        self,
        user_id: str,
        model_name: str,
        reward: float
    ):
        """
        更新模型性能
        
        Args:
            user_id: 用户ID
            model_name: 模型名称
            reward: 奖励值
        """
        stats = self.get_model_performance(user_id, model_name) or {
            "total_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": []
        }
        
        # 更新总计数
        stats["total_count"] += 1
        
        # 更新平均奖励
        old_avg = stats["avg_reward"]
        n = stats["total_count"]
        stats["avg_reward"] = (old_avg * (n - 1) + reward) / n
        
        # 更新最近奖励（保留最近20次）
        recent = stats.get("recent_rewards", [])
        recent.append(reward)
        recent = recent[-20:]  # 只保留最近20次
        
        with self._get_connection() as conn:
            existing = self.get_model_performance(user_id, model_name)
            
            if existing:
                conn.execute(
                    """
                    UPDATE model_performance 
                    SET total_count = ?, avg_reward = ?, recent_rewards = ?, last_used_at = ?
                    WHERE user_id = ? AND model_name = ?
                    """,
                    (
                        stats["total_count"],
                        stats["avg_reward"],
                        json.dumps(recent),
                        int(time.time()),
                        user_id,
                        model_name
                    )
                )
            else:
                conn.execute(
                    """
                    INSERT INTO model_performance 
                    (user_id, model_name, total_count, avg_reward, recent_rewards, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        model_name,
                        stats["total_count"],
                        stats["avg_reward"],
                        json.dumps(recent),
                        int(time.time()),
                        int(time.time())
                    )
                )
    
    # ============================================================
    # 缓存方法
    # ============================================================
    
    def get_cache(self, cache_key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            cache_key: 缓存键
        
        Returns:
            Any: 缓存值（JSON反序列化），过期或不存在返回None
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT result, expires_at FROM mcp_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row:
                expires_at = row["expires_at"]
                
                # 检查是否过期
                if int(time.time()) < expires_at:
                    return json.loads(row["result"])
                else:
                    # 过期，删除
                    self.delete_cache(cache_key)
            
            return None
    
    def set_cache(
        self,
        cache_key: str,
        tool_name: str,
        params_hash: str,
        result: Any,
        ttl: int = 300
    ):
        """
        设置缓存
        
        Args:
            cache_key: 缓存键
            tool_name: 工具名称
            params_hash: 参数哈希
            result: 缓存值（将JSON序列化）
            ttl: 过期时间（秒，默认300秒）
        """
        # 处理dataclass和其他不可直接序列化的对象
        from dataclasses import is_dataclass, asdict
        if is_dataclass(result):
            result_json = json.dumps(asdict(result))
        elif hasattr(result, '__dict__'):
            result_json = json.dumps(result.__dict__)
        else:
            try:
                result_json = json.dumps(result)
            except TypeError:
                # 如果还是无法序列化，转为字符串
                result_json = json.dumps(str(result))
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mcp_cache 
                (cache_key, tool_name, params_hash, result, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    tool_name,
                    params_hash,
                    result_json,
                    int(time.time()),
                    int(time.time()) + ttl
                )
            )
    
    def delete_cache(self, cache_key: str):
        """删除缓存"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM mcp_cache WHERE cache_key = ?", (cache_key,))
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM mcp_cache WHERE expires_at < ?",
                (int(time.time()),)
            )
            deleted_count = cursor.rowcount
            logger.info(f"✅ Cleaned up {deleted_count} expired cache entries")
            return deleted_count
    
    # ============================================================
    # 元学习统计方法（v2.0新增）
    # ============================================================
    
    def get_user_meta_learning_stats(self, user_id: str) -> Optional[Dict]:
        """
        获取小模型在该用户上下文的学习进度统计
        
        Args:
            user_id: 用户ID
        
        Returns:
            Dict: 学习进度统计，不存在返回None
                {
                    "user_id": "zhangsan",
                    "high_quality_samples": 65,
                    "total_samples": 100,
                    "current_phase": "transition",
                    "phase_start_time": 1698765432.0,
                    "last_update_time": 1698765432.0,
                    "student_model_success_rate": 0.85,
                    "teacher_model_success_rate": 0.92
                }
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_meta_learning_stats WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def update_user_meta_learning_stats(
        self,
        user_id: str,
        **updates
    ) -> bool:
        """
        更新小模型学习进度统计
        
        Args:
            user_id: 用户ID
            **updates: 要更新的字段
                high_quality_samples: int
                total_samples: int
                current_phase: str (teaching/transition/autonomous)
                phase_start_time: float
                student_model_success_rate: float
                teacher_model_success_rate: float
        
        Returns:
            bool: 更新成功返回True
        
        Example:
            db.update_user_meta_learning_stats(
                user_id="zhangsan",
                high_quality_samples=66,
                current_phase="transition"
            )
        """
        try:
            with self._get_connection() as conn:
                # 检查是否存在
                existing = self.get_user_meta_learning_stats(user_id)
                
                if existing:
                    # 更新现有记录
                    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                    set_clause += ", last_update_time = ?"
                    values = list(updates.values()) + [time.time(), user_id]
                    
                    conn.execute(
                        f"UPDATE user_meta_learning_stats SET {set_clause} WHERE user_id = ?",
                        values
                    )
                else:
                    # 插入新记录（初始化）
                    updates.update({
                        "user_id": user_id,
                        "phase_start_time": time.time(),
                        "last_update_time": time.time()
                    })
                    
                    columns = ", ".join(updates.keys())
                    placeholders = ", ".join("?" * len(updates))
                    
                    conn.execute(
                        f"INSERT INTO user_meta_learning_stats ({columns}) VALUES ({placeholders})",
                        list(updates.values())
                    )
                
                return True
        except Exception as e:
            logger.error(f"Failed to update meta learning stats for {user_id}: {e}")
            return False
    
    def increment_sample_count(
        self,
        user_id: str,
        is_high_quality: bool
    ):
        """
        增加样本计数（反馈时调用）
        
        Args:
            user_id: 用户ID
            is_high_quality: 是否高质量样本（reward >= 4.0）
        
        说明：
            - 自动更新high_quality_samples和total_samples
            - 自动检查并更新学习阶段（50→transition, 100→autonomous）
            - 线程安全（使用事务）
        """
        with self._get_connection() as conn:
            # 获取当前统计（带锁）
            cursor = conn.execute(
                "SELECT * FROM user_meta_learning_stats WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                stats = dict(row)
                
                # 更新计数
                stats["total_samples"] += 1
                if is_high_quality:
                    stats["high_quality_samples"] += 1
                
                # 检查阶段切换
                old_phase = stats["current_phase"]
                new_phase = old_phase
                
                if stats["high_quality_samples"] < 50:
                    new_phase = "teaching"
                elif stats["high_quality_samples"] < 100:
                    new_phase = "transition"
                else:
                    new_phase = "autonomous"
                
                # 阶段变化时更新phase_start_time
                if new_phase != old_phase:
                    stats["current_phase"] = new_phase
                    stats["phase_start_time"] = time.time()
                    logger.info(
                        f"🎯 User {user_id} phase changed: {old_phase} → {new_phase} "
                        f"(samples: {stats['high_quality_samples']})"
                    )
                
                # 原子更新
                conn.execute(
                    """
                    UPDATE user_meta_learning_stats 
                    SET high_quality_samples = ?,
                        total_samples = ?,
                        current_phase = ?,
                        phase_start_time = ?,
                        last_update_time = ?
                    WHERE user_id = ?
                    """,
                    (
                        stats["high_quality_samples"],
                        stats["total_samples"],
                        stats["current_phase"],
                        stats["phase_start_time"],
                        time.time(),
                        user_id
                    )
                )
            else:
                # 首次初始化
                conn.execute(
                    """
                    INSERT INTO user_meta_learning_stats 
                    (user_id, high_quality_samples, total_samples, current_phase, phase_start_time, last_update_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        1 if is_high_quality else 0,
                        1,
                        "teaching",
                        time.time(),
                        time.time()
                    )
                )
    
    def get_all_users_by_phase(self, phase: str) -> List[Dict]:
        """
        获取指定学习阶段的所有用户
        
        Args:
            phase: 学习阶段（teaching/transition/autonomous）
        
        Returns:
            List[Dict]: 用户列表，按high_quality_samples降序
        
        Example:
            # 获取所有处于过渡期的用户
            transition_users = db.get_all_users_by_phase("transition")
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM user_meta_learning_stats 
                WHERE current_phase = ?
                ORDER BY high_quality_samples DESC
                """,
                (phase,)
            )
            return [dict(row) for row in cursor.fetchall()]

