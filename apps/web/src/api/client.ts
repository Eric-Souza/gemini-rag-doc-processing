import type { ChatMode, Citation, DocumentItem, HealthResponse } from '../types'

const API_BASE = '/api'

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.detail === 'string') return data.detail
    return JSON.stringify(data?.detail ?? data)
  } catch {
    return res.statusText || 'Request failed'
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function listDocuments(): Promise<DocumentItem[]> {
  const res = await fetch(`${API_BASE}/documents`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/documents`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await parseError(res))
}

export type ChatStreamHandlers = {
  onToken: (text: string) => void
  onCitation: (citation: Citation) => void
  onDone: () => void
  onError: (message: string) => void
}

export async function streamChat(
  payload: { message: string; mode: ChatMode; document_ids?: string[] },
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })
  if (!res.ok) throw new Error(await parseError(res))
  if (!res.body) throw new Error('No response body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''

    for (const part of parts) {
      const line = part
        .split('\n')
        .map((l) => l.trim())
        .find((l) => l.startsWith('data:'))
      if (!line) continue
      const json = line.replace(/^data:\s*/, '')
      try {
        const event = JSON.parse(json) as {
          event: string
          text?: string
          message?: string
          document_id?: string
          filename?: string
          excerpt?: string
          score?: number | null
        }
        if (event.event === 'token' && event.text) handlers.onToken(event.text)
        else if (event.event === 'citation') {
          handlers.onCitation({
            document_id: event.document_id ?? '',
            filename: event.filename ?? 'unknown',
            excerpt: event.excerpt ?? '',
            score: event.score,
          })
        } else if (event.event === 'error') {
          handlers.onError(event.message ?? 'Chat error')
        } else if (event.event === 'done') {
          handlers.onDone()
        }
      } catch {
        // ignore malformed chunks
      }
    }
  }
}
