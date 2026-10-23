import { Alert, Box, Snackbar, Stack, Typography } from '@mui/material'
import { useCallback, useEffect, useState } from 'react'
import { deleteDocument, listDocuments, uploadDocument } from '../api/client'
import { DocumentTable } from '../components/DocumentTable'
import { UploadDropzone } from '../components/UploadDropzone'
import type { DocumentItem } from '../types'

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const docs = await listDocuments()
    setDocuments(docs)
  }, [])

  useEffect(() => {
    void refresh().catch((e: Error) => setError(e.message))
  }, [refresh])

  useEffect(() => {
    const indexing = documents.some(
      (d) => d.status === 'indexing' || d.status === 'pending',
    )
    if (!indexing) return
    const id = window.setInterval(() => {
      void refresh().catch(() => undefined)
    }, 2000)
    return () => window.clearInterval(id)
  }, [documents, refresh])

  const handleUpload = async (file: File) => {
    setUploading(true)
    setError(null)
    try {
      await uploadDocument(file)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (id: string) => {
    setError(null)
    try {
      await deleteDocument(id)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  return (
    <Stack spacing={3}>
      <Box>
        <Typography variant="h4" gutterBottom>
          Documents
        </Typography>
        <Typography color="text.secondary">
          Upload files to chunk, embed with Gemini, and store in Postgres + pgvector for grounded Q&A.
        </Typography>
      </Box>

      <UploadDropzone uploading={uploading} onUpload={handleUpload} />
      <DocumentTable documents={documents} onDelete={handleDelete} />

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Stack>
  )
}
