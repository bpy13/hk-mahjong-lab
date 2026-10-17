import { useEffect, useState } from 'react'

/** Colour theme: follow the system, or force light or dark. */
export type Theme = 'auto' | 'light' | 'dark'

const NEXT: Record<Theme, Theme> = { auto: 'dark', dark: 'light', light: 'auto' }
export const THEME_LABEL: Record<Theme, string> = { auto: '自動', light: '淺色', dark: '深色' }

function load(): Theme {
  try {
    const t = localStorage.getItem('theme')
    return t === 'light' || t === 'dark' ? t : 'auto'
  } catch {
    return 'auto'
  }
}

/** The current theme (applied as `data-theme` on <html>) and a function that
 * cycles auto → dark → light. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(load)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'auto') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('theme', theme)
    } catch {
      // storage unavailable: the theme just is not remembered
    }
  }, [theme])
  return [theme, () => setTheme((t) => NEXT[t])]
}

/** Tile scale for the window size, so the whole table fits without scrolling
 * (published as the CSS variable `--s`). */
export function useTileScale() {
  useEffect(() => {
    const update = () => {
      const s = Math.max(0.55, Math.min(1.35, (window.innerHeight - 20) / 880, (window.innerWidth - 360) / 1180))
      document.documentElement.style.setProperty('--s', s.toFixed(3))
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])
}
