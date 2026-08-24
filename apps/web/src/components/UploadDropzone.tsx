import CloudUploadOutlinedIcon from '@mui/icons-material/CloudUploadOutlined'
import { Box, LinearProgress, Typography } from '@mui/material'
import { motion } from 'framer-motion'
import { useRef, useState } from 'react'

const MotionPaper = motion.create(Box)

type Props = {
  uploading: boolean
  onUpload: (file: File) => Promise<void>
}

export function UploadDropzone({ uploading, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length || uploading) return
    await onUpload(files[0])
  }

  return (
    <MotionPaper
      animate={{
        scale: dragActive ? 1.01 : 1,
        borderColor: dragActive ? '#0F766E' : 'rgba(15, 23, 42, 0.16)',
      }}
      transition={{ type: 'spring', stiffness: 320, damping: 24 }}
      onDragEnter={(e) => {
        e.preventDefault()
        setDragActive(true)
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        setDragActive(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragActive(false)
        void handleFiles(e.dataTransfer.files)
      }}
      onClick={() => inputRef.current?.click()}
      sx={{
        p: 4,
        borderRadius: 3,
        border: '2px dashed',
        bgcolor: dragActive ? 'rgba(15, 118, 110, 0.06)' : 'background.paper',
        cursor: uploading ? 'wait' : 'pointer',
        textAlign: 'center',
        transition: 'background-color 0.2s ease',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain,text/markdown"
        onChange={(e) => void handleFiles(e.target.files)}
      />
      <CloudUploadOutlinedIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
      <Typography variant="h6" gutterBottom>
        Drop a document to index
      </Typography>
      <Typography color="text.secondary">
        PDF, TXT, or Markdown — click or drag and drop
      </Typography>
      {uploading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
        </Box>
      )}
    </MotionPaper>
  )
}
