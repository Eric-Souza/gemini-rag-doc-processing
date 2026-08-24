import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined'
import ForumOutlinedIcon from '@mui/icons-material/ForumOutlined'
import {
  AppBar,
  Box,
  Chip,
  Container,
  Tab,
  Tabs,
  Toolbar,
  Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { Link as RouterLink, Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { getHealth } from '../api/client'
import { PageTransition } from './PageTransition'

export function AppShell() {
  const location = useLocation()
  const tab = location.pathname.startsWith('/chat') ? 1 : 0
  const [geminiOk, setGeminiOk] = useState<boolean | null>(null)

  useEffect(() => {
    getHealth()
      .then((h) => setGeminiOk(h.gemini_configured))
      .catch(() => setGeminiOk(null))
  }, [])

  return (
    <Box sx={{ minHeight: '100vh', pb: 6 }}>
      <AppBar position="sticky">
        <Toolbar sx={{ gap: 2, flexWrap: 'wrap' }}>
          <Typography
            variant="h6"
            sx={{
              fontFamily: '"Source Serif 4", Georgia, serif',
              fontWeight: 600,
              letterSpacing: '-0.02em',
              mr: 1,
            }}
          >
            Doc Processing AI
          </Typography>
          <Tabs
            value={tab}
            textColor="primary"
            indicatorColor="primary"
            sx={{ minHeight: 48, '& .MuiTab-root': { minHeight: 48 } }}
          >
            <Tab
              icon={<DescriptionOutlinedIcon fontSize="small" />}
              iconPosition="start"
              label="Documents"
              component={RouterLink}
              to="/"
            />
            <Tab
              icon={<ForumOutlinedIcon fontSize="small" />}
              iconPosition="start"
              label="Chat"
              component={RouterLink}
              to="/chat"
            />
          </Tabs>
          <Box sx={{ flexGrow: 1 }} />
          {geminiOk !== null && (
            <Chip
              size="small"
              label={geminiOk ? 'Gemini ready' : 'Set GEMINI_API_KEY'}
              color={geminiOk ? 'success' : 'warning'}
              variant="outlined"
            />
          )}
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <AnimatePresence mode="wait">
          <PageTransition key={location.pathname}>
            <Outlet />
          </PageTransition>
        </AnimatePresence>
      </Container>
    </Box>
  )
}
