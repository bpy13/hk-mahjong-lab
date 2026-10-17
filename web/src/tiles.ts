// Tile indices follow docs/rules/hk.md §1: 1m–9m = 0–8, 1p–9p = 9–17,
// 1s–9s = 18–26, 東南西北 = 27–30, 白發中 = 31–33, 梅蘭菊竹 = 34–37, 春夏秋冬 = 38–41.

export const NUMERALS = '一二三四五六七八九'
const SUITS = '萬筒索'
const HONOURS = ['東', '南', '西', '北', '白', '發', '中']
const BONUS = ['梅', '蘭', '菊', '竹', '春', '夏', '秋', '冬']
export const WINDS = ['東', '南', '西', '北']

export type TileKind = 'suit' | 'wind' | 'dragon' | 'bonus'

/** Chinese name: 五筒, 東, 白, 蘭. */
export function tileName(t: number): string {
  if (t < 27) return NUMERALS[t % 9] + SUITS[Math.floor(t / 9)]
  if (t < 34) return HONOURS[t - 27]
  return BONUS[t - 34]
}

export function tileKind(t: number): TileKind {
  if (t < 27) return 'suit'
  if (t < 31) return 'wind'
  if (t < 34) return 'dragon'
  return 'bonus'
}

/** 0 = 萬, 1 = 筒, 2 = 索 for suited tiles, else -1. */
export function suitOf(t: number): number {
  return t < 27 ? Math.floor(t / 9) : -1
}

/** The two characters drawn on a tile face: a top glyph and an optional bottom one. */
export function faceText(t: number): { top: string; bottom: string } {
  if (t < 27) return { top: NUMERALS[t % 9], bottom: SUITS[Math.floor(t / 9)] }
  if (t < 34) return { top: HONOURS[t - 27], bottom: '' }
  return { top: BONUS[t - 34], bottom: String(((t - 34) % 4) + 1) }
}

/** Colour of a tile's characters. */
export function tileColour(t: number): string {
  if (t < 9) return '#b3261e' // 萬 red
  if (t < 18) return '#1f4e9c' // 筒 blue
  if (t < 27) return '#16703a' // 索 green
  if (t < 31) return '#222' // winds
  if (t === 31) return '#1f4e9c' // 白
  if (t === 32) return '#16703a' // 發
  if (t === 33) return '#b3261e' // 中
  return t < 38 ? '#8a3f9c' : '#c06a00' // flowers, seasons
}

/** Sorted copy (index order is the natural display order). */
export function sortTiles(tiles: number[]): number[] {
  return [...tiles].sort((a, b) => a - b)
}

// Action codes (mahjong_hk::Action::code).
export const SELF_DRAWN_WIN = 34
export const PASS = 103
export const WIN = 104
export const PUNG = 105
export const CLAIM_KONG = 106

export function isDiscard(a: number): boolean {
  return a < 34
}

/** Seat position relative to the human: 0 bottom, 1 right, 2 top, 3 left. */
export function relativeSeat(seat: number, human: number): number {
  return (seat - human + 4) % 4
}

/** A target plan in Chinese, e.g. 混一色（筒）· 對對糊 · 中. */
export function planName(target: {
  plan: string
  structure: string
  pungs: number[]
  limit: boolean
}): string {
  const suitName: Record<string, string> = { m: '萬', p: '筒', s: '索' }
  const parts: string[] = []
  if (target.plan.startsWith('half_flush')) parts.push(`混一色（${suitName[target.plan.slice(-1)]}）`)
  else if (target.plan.startsWith('full_flush')) parts.push(`清一色（${suitName[target.plan.slice(-1)]}）`)
  else if (target.plan === 'honours') parts.push('字一色')
  else if (target.plan === 'orphans') parts.push('十三么')
  if (target.structure === 'all_pungs') parts.push('對對糊')
  if (target.structure === 'all_chows') parts.push('平糊')
  for (const t of target.pungs) parts.push(tileName(t))
  if (parts.length === 0) parts.push('門前清及花')
  return parts.join(' · ')
}
