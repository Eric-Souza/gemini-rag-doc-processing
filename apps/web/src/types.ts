export type DocumentStatus = 'pending' | 'indexing' | 'ready' | 'failed'

export interface DocumentItem {
  id: string
  filename: string
  content_type: string
  size_bytes: number
  status: DocumentStatus
  error_message: string | null
  chunk_count: number
  created_at: string
  updated_at: string
}

export type ChatMode = 'rag' | 'agent'

export interface Citation {
  document_id: string
  filename: string
  excerpt: string
  score?: number | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
  status?: string
}

export interface HealthResponse {
  status: string
  gemini_configured: boolean
  langsmith_configured?: boolean
  chat_model?: string
  embed_model?: string
}
