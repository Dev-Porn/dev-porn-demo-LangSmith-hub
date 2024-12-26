# dev-porn-demo-LangSmith-hub

# 基於 LLM 的檢索增強生成 (RAG) 與評估

專案實現了使用 LangChain 和 LangSmith 的 **檢索增強生成 (RAG)** 以及 **基於 LLM 的評估**。系統透過檢索相關文檔生成答案，並根據預定義的標準評估輸出質量。

## 功能特色

- **RAG 工作流程**：
  - 使用向量存儲檢索相關文檔。
  - 根據檢索到的上下文利用語言模型生成答案。
  - 以 LangChain 元件構建靈活的工作流程。

- **評估工作流程**：
  - 使用評估提示評估生成的答案質量。
  - 根據預定義的標準（如相關性和準確性）對輸出進行評分。

- **LangSmith 集成**：
  - 可視化並追蹤 RAG 與評估的工作流程。

- **Hub 提供的 Prompt 實用性**：
  - **適配性高**：直接從 LangChain Hub 拉取經驗證的 Prompt，滿足 RAG 與評估的需求。
  - **模組化設計**：支援靈活整合，並能適應不同的上下文和使用場景。


## Hub Prompt 的應用

專案從 LangChain Hub 拉取了以下兩個 Prompt，分別用於實現 RAG 和評估的核心邏輯：
1. **RAG Prompt**（`daethyra/rag-prompt`）：幫助系統根據檢索上下文生成準確且相關的答案。
2. **評估 Prompt**（`jisujiji/rag-prompt`）：專注於評估生成的答案質量，根據提供的標準給出分數。

### Hub Prompt 的優勢
- **即插即用**：不需要從零設計 Prompt，直接利用社群驗證的高效模版。
- **靈活擴展**：Prompt 可根據特定應用場景進行輕微調整以適應需求。
- **準確性**：專為特定任務設計，確保生成的答案與評估結果更準確。

---

![RAG Trace](/Users/wuhaosheng/dev/langsmith-hub/resource/rag-flow.png)
![RAG Trace](/Users/wuhaosheng/dev/langsmith-hub/resource/evaluation.png)