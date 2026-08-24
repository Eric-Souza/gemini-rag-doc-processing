import { Chip, Stack, Tooltip } from '@mui/material'
import { motion } from 'framer-motion'
import type { Citation } from '../types'

const MotionChip = motion.create(Chip)

type Props = {
  citations: Citation[]
}

export function CitationList({ citations }: Props) {
  if (!citations.length) return null

  return (
    <Stack
      direction="row"
      spacing={1}
      useFlexGap
      sx={{ mt: 1.5, flexWrap: 'wrap' }}
    >
      {citations.map((c, i) => (
        <Tooltip
          key={`${c.document_id}-${i}`}
          title={c.excerpt}
          arrow
          placement="top"
        >
          <MotionChip
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.25 }}
            size="small"
            label={c.filename}
            variant="outlined"
            color="primary"
          />
        </Tooltip>
      ))}
    </Stack>
  )
}
