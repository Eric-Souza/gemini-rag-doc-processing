import { Box } from '@mui/material'
import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

const MotionBox = motion.create(Box)

export function PageTransition({ children }: { children: ReactNode }) {
  return (
    <MotionBox
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
    >
      {children}
    </MotionBox>
  )
}
