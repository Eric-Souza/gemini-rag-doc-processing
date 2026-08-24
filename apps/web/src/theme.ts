import { createTheme } from '@mui/material/styles'

const ink = '#0F172A'
const teal = '#0F766E'
const tealDark = '#115E59'
const paper = '#F1F5F9'
const surface = '#FFFFFF'

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: teal,
      dark: tealDark,
      light: '#14B8A6',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#334155',
    },
    background: {
      default: paper,
      paper: surface,
    },
    text: {
      primary: ink,
      secondary: '#475569',
    },
    success: { main: teal },
    warning: { main: '#D97706' },
    error: { main: '#DC2626' },
    divider: 'rgba(15, 23, 42, 0.08)',
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
    h1: {
      fontFamily: '"Source Serif 4", Georgia, serif',
      fontWeight: 600,
    },
    h2: {
      fontFamily: '"Source Serif 4", Georgia, serif',
      fontWeight: 600,
    },
    h3: {
      fontFamily: '"Source Serif 4", Georgia, serif',
      fontWeight: 600,
    },
    h4: {
      fontFamily: '"Source Serif 4", Georgia, serif',
      fontWeight: 600,
    },
    h5: {
      fontFamily: '"Source Serif 4", Georgia, serif',
      fontWeight: 600,
    },
    h6: {
      fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
      fontWeight: 700,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          minHeight: '100vh',
          backgroundImage:
            'linear-gradient(145deg, #E2E8F0 0%, #F1F5F9 42%, #ECFEFF 100%)',
          backgroundAttachment: 'fixed',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
        },
      },
    },
    MuiPaper: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: {
          border: '1px solid rgba(15, 23, 42, 0.08)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255,255,255,0.86)',
          color: ink,
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(15, 23, 42, 0.08)',
          boxShadow: 'none',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
  },
})
