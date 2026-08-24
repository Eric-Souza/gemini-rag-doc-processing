import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined'
import {
  Chip,
  IconButton,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material'
import { motion } from 'framer-motion'
import type { DocumentItem, DocumentStatus } from '../types'

const MotionRow = motion.create(TableRow)

const statusColor: Record<
  DocumentStatus,
  'default' | 'warning' | 'success' | 'error'
> = {
  pending: 'default',
  indexing: 'warning',
  ready: 'success',
  failed: 'error',
}

type Props = {
  documents: DocumentItem[]
  onDelete: (id: string) => Promise<void>
}

export function DocumentTable({ documents, onDelete }: Props) {
  if (!documents.length) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          No documents yet
        </Typography>
        <Typography color="text.secondary">
          Upload a file above to start building your knowledge base.
        </Typography>
      </Paper>
    )
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Filename</TableCell>
            <TableCell>Status</TableCell>
            <TableCell align="right">Chunks</TableCell>
            <TableCell align="right">Size</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {documents.map((doc, index) => (
            <MotionRow
              key={doc.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.04, duration: 0.25 }}
              hover
            >
              <TableCell>
                <Stack spacing={0.5}>
                  <Typography sx={{ fontWeight: 600 }}>{doc.filename}</Typography>
                  {doc.error_message && (
                    <Typography variant="caption" color="error">
                      {doc.error_message}
                    </Typography>
                  )}
                </Stack>
              </TableCell>
              <TableCell>
                <Chip
                  size="small"
                  label={doc.status}
                  color={statusColor[doc.status]}
                  variant={doc.status === 'ready' ? 'filled' : 'outlined'}
                />
              </TableCell>
              <TableCell align="right">{doc.chunk_count}</TableCell>
              <TableCell align="right">
                {(doc.size_bytes / 1024).toFixed(1)} KB
              </TableCell>
              <TableCell align="right">
                <Tooltip title="Delete">
                  <IconButton
                    aria-label={`Delete ${doc.filename}`}
                    onClick={() => void onDelete(doc.id)}
                    color="error"
                    size="small"
                  >
                    <DeleteOutlinedIcon />
                  </IconButton>
                </Tooltip>
              </TableCell>
            </MotionRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}
