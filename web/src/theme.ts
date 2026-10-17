import type { RefObject } from 'react'
import { useEffect, useLayoutEffect, useState } from 'react'

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

/** Tile scale (the CSS variable `--s`) so the whole table fits the window
 * without scrolling: start from the window size, then shrink until the table's
 * content no longer overflows it. Re-fits on resize and when `key` changes. */
export function useTileScale(table: RefObject<HTMLElement | null>, key: unknown) {
  useLayoutEffect(() => {
    const set = (s: number) => document.documentElement.style.setProperty('--s', s.toFixed(3))
    const fit = () => {
      let s = Math.max(0.5, Math.min(1.35, (window.innerHeight - 20) / 880, (window.innerWidth - 360) / 1180))
      set(s)
      const el = table.current
      if (!el) return
      // Reading the size forces a layout, so each step sees the new scale.
      while ((el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1) && s > 0.4) {
        s -= 0.02
        set(s)
      }
    }
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [table, key])
}
