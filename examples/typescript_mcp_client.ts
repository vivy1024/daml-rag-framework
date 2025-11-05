/**
 * 玉珍健身 MCP TypeScript客户端示例
 * 展示前端应用如何与玉珍健身 MCP服务器集成

这个示例演示了：
1. TypeScript类型的MCP客户端
2. React/Vue等前端框架的集成
3. 错误处理和重试机制
4. 实时查询和反馈收集
5. 前端状态管理

作者：薛小川 (Xue Xiaochuan)
版本：v1.0.0
日期：2025-11-05
 */

// ============================================================================
// 类型定义
// ============================================================================

interface MCPClientConfig {
  baseUrl: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

interface QueryRequest {
  query: string;
  domain?: string;
  userId?: string;
  sessionId?: string;
  topK?: number;
  filters?: Record<string, any>;
}

interface QueryResponse {
  answer: string;
  sources: Array<{
    content: string;
    metadata: Record<string, any>;
    score: number;
  }>;
  retrievalMetadata: Record<string, any>;
  executionTime: number;
  modelUsed: string;
}

interface FeedbackRequest {
  sessionId: string;
  query: string;
  answer: string;
  userRating: number; // 1-5
  userFeedback?: string;
  improvementSuggestions?: string;
}

interface ToolInfo {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

interface MCPHealthStatus {
  status: 'healthy' | 'unhealthy';
  frameworkHealth: Record<string, any>;
  toolsAvailable: string[];
}

// ============================================================================
// MCP客户端类
// ============================================================================

class DAMLRAGMCPClient {
  private config: MCPClientConfig;
  private baseURL: string;

  constructor(config: MCPClientConfig) {
    this.config = {
      timeout: 30000,
      retryAttempts: 3,
      retryDelay: 1000,
      ...config
    };
    this.baseURL = this.config.baseUrl.replace(/\/$/, ''); // 移除末尾斜杠
  }

  // ========================================================================
  // 私有方法
  // ========================================================================

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: AbortSignal.timeout(this.config.timeout!),
    };

    let lastError: Error;

    for (let attempt = 0; attempt < this.config.retryAttempts!; attempt++) {
      try {
        const response = await fetch(url, { ...defaultOptions, ...options });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        return await response.json();
      } catch (error) {
        lastError = error as Error;

        if (attempt === this.config.retryAttempts! - 1) {
          break;
        }

        // 指数退避
        const delay = this.config.retryDelay! * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError!;
  }

  // ========================================================================
  // 基础API方法
  // ========================================================================

  /**
   * 检查服务器健康状态
   */
  async checkHealth(): Promise<MCPHealthStatus> {
    return this.makeRequest<MCPHealthStatus>('/health');
  }

  /**
   * 列出所有可用工具
   */
  async listTools(): Promise<{ tools: ToolInfo[] }> {
    return this.makeRequest<{ tools: ToolInfo[] }>('/tools');
  }

  /**
   * 获取服务器统计信息
   */
  async getStatistics(): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('/statistics');
  }

  // ========================================================================
  // 核心功能方法
  // ========================================================================

  /**
   * 执行智能问答查询
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.makeRequest<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({
        query: request.query,
        domain: request.domain || 'general',
        user_id: request.userId,
        session_id: request.sessionId,
        top_k: request.topK || 10,
        filters: request.filters || {}
      })
    });
  }

  /**
   * 使用特定MCP工具
   */
  async useTool(toolName: string, parameters: Record<string, any>): Promise<any> {
    return this.makeRequest<any>(`/tools/${toolName}`, {
      method: 'POST',
      body: JSON.stringify(parameters)
    });
  }

  /**
   * 提交用户反馈
   */
  async submitFeedback(feedback: FeedbackRequest): Promise<{
    status: string;
    message: string;
    feedbackId: number;
  }> {
    return this.makeRequest('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: feedback.sessionId,
        query: feedback.query,
        answer: feedback.answer,
        user_rating: feedback.userRating,
        user_feedback: feedback.userFeedback,
        improvement_suggestions: feedback.improvementSuggestions
      })
    });
  }

  // ========================================================================
  // 便捷方法
  // ========================================================================

  /**
   * 智能问答便捷方法
   */
  async intelligentQA(
    query: string,
    domain: string = 'general',
    userId?: string
  ): Promise<QueryResponse> {
    return this.useTool('intelligent_qa', {
      query,
      domain,
      user_id: userId
    });
  }

  /**
   * 文档检索便捷方法
   */
  async documentRetrieval(
    query: string,
    retrievalMethod: string = 'three_tier',
    topK: number = 10
  ): Promise<any> {
    return this.useTool('document_retrieval', {
      query,
      retrieval_method: retrievalMethod,
      top_k: topK
    });
  }

  /**
   * 知识图谱查询便捷方法
   */
  async knowledgeGraphQuery(
    entities: string[],
    relationshipTypes?: string[],
    maxDepth: number = 2
  ): Promise<any> {
    return this.useTool('knowledge_graph_query', {
      entities,
      relationship_types: relationshipTypes || [],
      max_depth: maxDepth
    });
  }

  /**
   * 个性化推荐便捷方法
   */
  async personalizedRecommendation(
    userId: string,
    recommendationType: string = 'general',
    context?: Record<string, any>
  ): Promise<any> {
    return this.useTool('personalized_recommendation', {
      user_id: userId,
      recommendation_type: recommendationType,
      context: context || {}
    });
  }

  /**
   * 质量评估便捷方法
   */
  async qualityAssessment(
    query: string,
    answer: string,
    sources: Array<{ content: string; metadata?: Record<string, any> }>
  ): Promise<any> {
    return this.useTool('quality_assessment', {
      query,
      answer,
      sources
    });
  }
}

// ============================================================================
// React Hook示例
// ============================================================================

import { useState, useEffect, useCallback } from 'react';

/**
 * React Hook: 使用MCP客户端
 */
export function useDAMLRRAGMCP(config: MCPClientConfig) {
  const [client] = useState(() => new DAMLRAGMCPClient(config));
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableTools, setAvailableTools] = useState<string[]>([]);

  // 检查连接状态
  useEffect(() => {
    const checkConnection = async () => {
      try {
        setIsLoading(true);
        const health = await client.checkHealth();
        setIsConnected(health.status === 'healthy');
        setAvailableTools(health.toolsAvailable);
        setError(null);
      } catch (err) {
        setIsConnected(false);
        setError(err instanceof Error ? err.message : '连接失败');
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
  }, [client]);

  // 智能问答
  const query = useCallback(async (request: QueryRequest) => {
    if (!isConnected) {
      throw new Error('MCP客户端未连接');
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await client.query(request);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '查询失败';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [client, isConnected]);

  // 提交反馈
  const submitFeedback = useCallback(async (feedback: FeedbackRequest) => {
    if (!isConnected) {
      throw new Error('MCP客户端未连接');
    }

    try {
      const result = await client.submitFeedback(feedback);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '反馈提交失败';
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  }, [client, isConnected]);

  return {
    client,
    isConnected,
    isLoading,
    error,
    availableTools,
    query,
    submitFeedback
  };
}

// ============================================================================
// Vue Composable示例
// ============================================================================

import { ref, computed, onMounted } from 'vue';

/**
 * Vue Composable: 使用MCP客户端
 */
export function useDAMLRRAGMCPVue(config: MCPClientConfig) {
  const client = new DAMLRAGMCPClient(config);
  const isConnected = ref(false);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const availableTools = ref<string[]>([]);

  const isReady = computed(() => isConnected.value && !isLoading.value);

  // 检查连接状态
  const checkConnection = async () => {
    try {
      isLoading.value = true;
      const health = await client.checkHealth();
      isConnected.value = health.status === 'healthy';
      availableTools.value = health.toolsAvailable;
      error.value = null;
    } catch (err) {
      isConnected.value = false;
      error.value = err instanceof Error ? err.message : '连接失败';
    } finally {
      isLoading.value = false;
    }
  };

  // 智能问答
  const query = async (request: QueryRequest): Promise<QueryResponse> => {
    if (!isConnected.value) {
      throw new Error('MCP客户端未连接');
    }

    isLoading.value = true;
    error.value = null;

    try {
      const result = await client.query(request);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '查询失败';
      error.value = errorMessage;
      throw new Error(errorMessage);
    } finally {
      isLoading.value = false;
    }
  };

  // 提交反馈
  const submitFeedback = async (feedback: FeedbackRequest) => {
    if (!isConnected.value) {
      throw new Error('MCP客户端未连接');
    }

    try {
      const result = await client.submitFeedback(feedback);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '反馈提交失败';
      error.value = errorMessage;
      throw new Error(errorMessage);
    }
  };

  onMounted(() => {
    checkConnection();
  });

  return {
    client,
    isConnected: computed(() => isConnected.value),
    isLoading: computed(() => isLoading.value),
    error: computed(() => error.value),
    availableTools: computed(() => availableTools.value),
    isReady,
    query,
    submitFeedback,
    checkConnection
  };
}

// ============================================================================
// 使用示例
// ============================================================================

/**
 * 基础使用示例
 */
async function basicUsageExample() {
  console.log('🔥 TypeScript MCP客户端基础使用示例');

  const client = new DAMLRAGMCPClient({
    baseUrl: 'http://localhost:8002',
    timeout: 30000,
    retryAttempts: 3
  });

  try {
    // 1. 检查服务器状态
    const health = await client.checkHealth();
    console.log(`📊 服务器状态: ${health.status}`);
    console.log(`🛠️  可用工具: ${health.toolsAvailable.join(', ')}`);

    // 2. 执行智能问答
    const query = "初学者如何制定健身计划？";
    const result = await client.intelligentQA(query, 'fitness', 'demo_user');

    console.log(`❓ 查询: ${query}`);
    console.log(`🤖 回答: ${result.answer.substring(0, 200)}...`);
    console.log(`📚 来源数量: ${result.sources.length}`);
    console.log(`⏱️  执行时间: ${result.executionTime}秒`);

    // 3. 提交反馈
    const feedbackResult = await client.submitFeedback({
      sessionId: 'session_123',
      query: query,
      answer: result.answer,
      userRating: 5,
      userFeedback: '回答很有帮助！'
    });

    console.log(`✅ 反馈提交: ${feedbackResult.message}`);

  } catch (error) {
    console.error('❌ 示例运行失败:', error);
  }
}

/**
 * React组件示例
 */
function FitnessQAComponent() {
  const {
    isConnected,
    isLoading,
    error,
    query,
    submitFeedback
  } = useDAMLRRAGMCP({
    baseUrl: 'http://localhost:8002'
  });

  const [currentQuery, setCurrentQuery] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [userRating, setUserRating] = useState(5);

  const handleQuery = async () => {
    if (!currentQuery.trim()) return;

    try {
      const result = await query({
        query: currentQuery,
        domain: 'fitness',
        userId: 'react_user'
      });

      setCurrentAnswer(result.answer);
    } catch (error) {
      console.error('查询失败:', error);
    }
  };

  const handleFeedback = async () => {
    if (!currentQuery || !currentAnswer) return;

    try {
      await submitFeedback({
        sessionId: 'react_session',
        query: currentQuery,
        answer: currentAnswer,
        userRating
      });

      alert('反馈已提交，感谢您的评价！');
    } catch (error) {
      console.error('反馈提交失败:', error);
    }
  };

  if (!isConnected) {
    return <div>正在连接MCP服务器...</div>;
  }

  return (
    <div className="fitness-qa">
      <h2>健身智能问答</h2>

      <div className="query-input">
        <input
          type="text"
          value={currentQuery}
          onChange={(e) => setCurrentQuery(e.target.value)}
          placeholder="请输入健身相关问题..."
          disabled={isLoading}
        />
        <button onClick={handleQuery} disabled={isLoading}>
          {isLoading ? '查询中...' : '提问'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {currentAnswer && (
        <div className="answer">
          <h3>回答：</h3>
          <p>{currentAnswer}</p>

          <div className="feedback">
            <h4>请评价这个回答：</h4>
            <select
              value={userRating}
              onChange={(e) => setUserRating(Number(e.target.value))}
            >
              <option value={5}>非常满意</option>
              <option value={4}>满意</option>
              <option value={3}>一般</option>
              <option value={2}>不满意</option>
              <option value={1}>非常不满意</option>
            </select>
            <button onClick={handleFeedback}>提交反馈</button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Vue组件示例
 */
const FitnessQAComponent = {
  setup() {
    const {
      isConnected,
      isLoading,
      error,
      query,
      submitFeedback
    } = useDAMLRRAGMCPVue({
      baseUrl: 'http://localhost:8002'
    });

    const currentQuery = ref('');
    const currentAnswer = ref('');
    const userRating = ref(5);

    const handleQuery = async () => {
      if (!currentQuery.value.trim()) return;

      try {
        const result = await query.value({
          query: currentQuery.value,
          domain: 'fitness',
          userId: 'vue_user'
        });

        currentAnswer.value = result.answer;
      } catch (error) {
        console.error('查询失败:', error);
      }
    };

    const handleFeedback = async () => {
      if (!currentQuery.value || !currentAnswer.value) return;

      try {
        await submitFeedback.value({
          sessionId: 'vue_session',
          query: currentQuery.value,
          answer: currentAnswer.value,
          userRating: userRating.value
        });

        alert('反馈已提交，感谢您的评价！');
      } catch (error) {
        console.error('反馈提交失败:', error);
      }
    };

    return {
      isConnected,
      isLoading,
      error,
      currentQuery,
      currentAnswer,
      userRating,
      handleQuery,
      handleFeedback
    };
  },

  template: `
    <div class="fitness-qa">
      <h2>健身智能问答</h2>

      <div v-if="!isConnected">正在连接MCP服务器...</div>

      <div v-else>
        <div class="query-input">
          <input
            v-model="currentQuery"
            type="text"
            placeholder="请输入健身相关问题..."
            :disabled="isLoading"
          />
          <button @click="handleQuery" :disabled="isLoading">
            {{ isLoading ? '查询中...' : '提问' }}
          </button>
        </div>

        <div v-if="error" class="error">{{ error }}</div>

        <div v-if="currentAnswer" class="answer">
          <h3>回答：</h3>
          <p>{{ currentAnswer }}</p>

          <div class="feedback">
            <h4>请评价这个回答：</h4>
            <select v-model="userRating">
              <option :value="5">非常满意</option>
              <option :value="4">满意</option>
              <option :value="3">一般</option>
              <option :value="2">不满意</option>
              <option :value="1">非常不满意</option>
            </select>
            <button @click="handleFeedback">提交反馈</button>
          </div>
        </div>
      </div>
    </div>
  `
};

// ============================================================================
// 导出
// ============================================================================

export {
  DAMLRAGMCPClient,
  type MCPClientConfig,
  type QueryRequest,
  type QueryResponse,
  type FeedbackRequest,
  type ToolInfo,
  type MCPHealthStatus
};

// 如果是直接运行此文件，执行示例
if (typeof window !== 'undefined') {
  // 浏览器环境
  console.log('玉珍健身 MCP TypeScript客户端已加载');

  // 可以在这里初始化全局客户端实例
  window.damlragMCPClient = new DAMLRAGMCPClient({
    baseUrl: process.env.NODE_ENV === 'production'
      ? 'https://your-production-server.com'
      : 'http://localhost:8002'
  });
} else if (typeof module !== 'undefined' && module.exports) {
  // Node.js环境
  console.log('玉珍健身 MCP TypeScript客户端 (Node.js版本)');

  // 运行基础示例
  basicUsageExample().catch(console.error);
}