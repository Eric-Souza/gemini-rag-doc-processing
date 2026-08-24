import { Box, CircularProgress, Stack, Typography } from '@mui/material'
import { keyframes } from '@mui/system'
import { AnimatePresence, motion } from 'framer-motion'
import type { ChatMessage } from '../types'
import { CitationList } from './CitationList'

const pulse = keyframes`
  0%, 100% { opacity: 0.2; }
  50% { opacity: 1; }
`

const MotionBox = motion.create(Box)

type Props = {
  message: ChatMessage
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const showStatus = Boolean(message.streaming && message.status)

  return (
    <MotionBox
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Box
        sx={{
          maxWidth: '78%',
          px: 2,
          py: 1.5,
          borderRadius: 3,
          bgcolor: isUser ? 'primary.main' : 'background.paper',
          color: isUser ? 'primary.contrastText' : 'text.primary',
          border: isUser ? 'none' : '1px solid',
          borderColor: 'divider',
        }}
      >
        <AnimatePresence mode="wait">
          {showStatus && (
            <MotionBox
              key={message.status}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2 }}
              sx={{ mb: message.content ? 1 : 0 }}
            >
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                <CircularProgress size={14} thickness={5} color="primary" />
                <Typography variant="body2" color="text.secondary">
                  {message.status}
                </Typography>
              </Stack>
            </MotionBox>
          )}
        </AnimatePresence>

        {(message.content || (message.streaming && !showStatus)) && (
          <Typography
            variant="body1"
            sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
          >
            {message.content || (message.streaming ? '' : '…')}
            {message.streaming && message.content && (
              <Box
                component="span"
                sx={{
                  display: 'inline-block',
                  width: 8,
                  height: '1.1em',
                  ml: 0.5,
                  bgcolor: 'primary.main',
                  verticalAlign: 'text-bottom',
                  animation: `${pulse} 1s ease-in-out infinite`,
                }}
              />
            )}
          </Typography>
        )}
        {!isUser && message.citations && (
          <CitationList citations={message.citations} />
        )}
      </Box>
    </MotionBox>
  )
}
