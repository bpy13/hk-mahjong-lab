import { NUMERALS } from '../tiles'

// Tile faces drawn as on real tiles, in a 36 × 48 box: dots as circles, bamboo as
// sticks (the 1 of bamboo as a bird), characters as a numeral over 萬, honours as
// their character, and flowers and seasons as small pictures with their number.

const BLUE = '#1f4e9c'
const GREEN = '#16703a'
const RED = '#b3261e'
const INK = '#1b2a5c'
const SERIF = "'Noto Serif TC', 'Songti TC', serif"

type Spot = [x: number, y: number, colour: string, angle?: number]

/** Circle centres and colours for 1–9 dots; the radius is per count. */
export const DOTS: { r: number; spots: Spot[] }[] = [
  { r: 12, spots: [[18, 24, RED]] },
  { r: 7.5, spots: [[18, 14, GREEN], [18, 34, BLUE]] },
  { r: 5.5, spots: [[10, 12, BLUE], [18, 24, RED], [26, 36, GREEN]] },
  { r: 6, spots: [[11, 15, BLUE], [25, 15, GREEN], [11, 33, GREEN], [25, 33, BLUE]] },
  { r: 5, spots: [[10, 12, BLUE], [26, 12, GREEN], [18, 24, RED], [10, 36, GREEN], [26, 36, BLUE]] },
  {
    r: 5,
    spots: [[11, 10, GREEN], [25, 10, GREEN], [11, 26, RED], [25, 26, RED], [11, 38, RED], [25, 38, RED]],
  },
  {
    r: 4.3,
    spots: [
      [9, 8, GREEN], [18, 12, GREEN], [27, 16, GREEN],
      [12, 28, RED], [24, 28, RED], [12, 39, RED], [24, 39, RED],
    ],
  },
  {
    r: 4.3,
    spots: [
      [12, 9, BLUE], [24, 9, BLUE], [12, 19, BLUE], [24, 19, BLUE],
      [12, 29, BLUE], [24, 29, BLUE], [12, 39, BLUE], [24, 39, BLUE],
    ],
  },
  {
    r: 4.3,
    spots: [
      [9, 12, BLUE], [18, 12, BLUE], [27, 12, BLUE],
      [9, 24, RED], [18, 24, RED], [27, 24, RED],
      [9, 36, GREEN], [18, 36, GREEN], [27, 36, GREEN],
    ],
  },
]

/** Stick centres and colours for 2–9 bamboo (index 0, the bird, is drawn apart);
 * the stick height is per count. */
export const STICKS: { h: number; spots: Spot[] }[] = [
  { h: 0, spots: [] },
  { h: 16, spots: [[18, 14, GREEN], [18, 34, BLUE]] },
  { h: 16, spots: [[18, 14, BLUE], [12, 34, GREEN], [24, 34, GREEN]] },
  { h: 16, spots: [[12, 14, BLUE], [24, 14, GREEN], [12, 34, GREEN], [24, 34, BLUE]] },
  { h: 15, spots: [[10, 13, GREEN], [26, 13, BLUE], [18, 24, RED], [10, 35, BLUE], [26, 35, GREEN]] },
  {
    h: 16,
    spots: [[10, 14, GREEN], [18, 14, GREEN], [26, 14, GREEN], [10, 34, BLUE], [18, 34, BLUE], [26, 34, BLUE]],
  },
  {
    h: 11,
    spots: [
      [18, 10, RED],
      [10, 25, GREEN], [18, 25, GREEN], [26, 25, GREEN],
      [10, 38, GREEN], [18, 38, GREEN], [26, 38, GREEN],
    ],
  },
  {
    // An M over a W: outer sticks upright, inner ones leaning.
    h: 15,
    spots: [
      [7, 14, GREEN], [14.5, 14, GREEN, -20], [21.5, 14, GREEN, 20], [29, 14, GREEN],
      [7, 34, GREEN], [14.5, 34, GREEN, 20], [21.5, 34, GREEN, -20], [29, 34, GREEN],
    ],
  },
  {
    h: 11,
    spots: [
      [10, 10, GREEN], [18, 10, RED], [26, 10, BLUE],
      [10, 24, GREEN], [18, 24, RED], [26, 24, BLUE],
      [10, 38, GREEN], [18, 38, RED], [26, 38, BLUE],
    ],
  },
]

function Dot({ x, y, r, colour }: { x: number; y: number; r: number; colour: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={colour} />
      <circle cx={x} cy={y} r={r * 0.68} fill="#fbf7ec" />
      <circle cx={x} cy={y} r={r * 0.45} fill={colour} />
      {r > 8 && <circle cx={x} cy={y} r={r * 0.22} fill="#fbf7ec" />}
    </g>
  )
}

function Stick({ x, y, h, colour, angle = 0 }: { x: number; y: number; h: number; colour: string; angle?: number }) {
  return (
    <g transform={angle ? `rotate(${angle} ${x} ${y})` : undefined}>
      <rect x={x - 2} y={y - h / 2} width="4" height={h} rx="2" fill={colour} />
      <line x1={x - 2} x2={x + 2} y1={y} y2={y} stroke="#fbf7ec" strokeWidth="1" />
      <line x1={x} x2={x} y1={y - h / 2 + 2} y2={y + h / 2 - 2} stroke="#fbf7ec" strokeWidth="0.6" opacity="0.6" />
    </g>
  )
}

/** The 1 of bamboo: a bird on a stem. */
function Bird() {
  return (
    <g>
      <path d="M12 30 C 4 26, 6 14, 13 18 C 16 20, 15 26, 12 30 Z" fill={BLUE} />
      <path d="M12 30 C 4 34, 4 42, 10 40" fill="none" stroke={GREEN} strokeWidth="2.5" strokeLinecap="round" />
      <ellipse cx="19" cy="27" rx="8" ry="6" fill={GREEN} />
      <circle cx="25" cy="17" r="4.5" fill={GREEN} />
      <circle cx="26" cy="16" r="1.2" fill="#fbf7ec" />
      <path d="M29 16 L33 18 L29 19 Z" fill={RED} />
      <path d="M22 13 L24 8 L26 13" fill={RED} />
      <path d="M17 33 L16 40 M21 33 L22 40" stroke={RED} strokeWidth="1.5" strokeLinecap="round" />
    </g>
  )
}

/** Small pictures for 梅蘭菊竹 (0–3) and 春夏秋冬 (4–7). */
function Bonus({ k }: { k: number }) {
  const n = (k % 4) + 1
  const numberColour = k < 4 ? RED : BLUE
  let picture
  switch (k) {
    case 0: // 梅 plum blossom on a branch
      picture = (
        <g>
          <path d="M8 42 C 14 34, 16 30, 26 22" stroke="#6b4a2b" strokeWidth="2" fill="none" strokeLinecap="round" />
          {[0, 1, 2, 3, 4].map((i) => {
            const a = (i * 72 - 90) * (Math.PI / 180)
            return <circle key={i} cx={20 + 5 * Math.cos(a)} cy={22 + 5 * Math.sin(a)} r="4" fill="#e0607e" />
          })}
          <circle cx="20" cy="22" r="2.5" fill="#f6c64f" />
        </g>
      )
      break
    case 1: // 蘭 orchid leaves and a flower
      picture = (
        <g fill="none" stroke={GREEN} strokeWidth="2" strokeLinecap="round">
          <path d="M18 42 C 16 32, 10 26, 6 20" />
          <path d="M18 42 C 20 30, 26 26, 30 24" />
          <path d="M18 42 C 18 34, 18 28, 20 20" />
          <g stroke="none" fill="#8a3f9c">
            <ellipse cx="20" cy="15" rx="3" ry="5" />
            <ellipse cx="16" cy="18" rx="4" ry="2.5" />
            <ellipse cx="24" cy="18" rx="4" ry="2.5" />
          </g>
        </g>
      )
      break
    case 2: // 菊 chrysanthemum
      picture = (
        <g>
          <path d="M18 40 L18 32" stroke={GREEN} strokeWidth="2" />
          {Array.from({ length: 12 }, (_, i) => (
            <ellipse key={i} cx="18" cy="16" rx="2" ry="7" fill="#e89a1a" transform={`rotate(${i * 30} 18 23)`} />
          ))}
          <circle cx="18" cy="23" r="3" fill="#b3261e" />
        </g>
      )
      break
    case 3: // 竹 bamboo stalk with leaves
      picture = (
        <g>
          <rect x="15" y="10" width="5" height="32" rx="2" fill={GREEN} />
          {[18, 26, 34].map((y) => (
            <line key={y} x1="14" x2="21" y1={y} y2={y} stroke="#fbf7ec" strokeWidth="1.2" />
          ))}
          <path d="M20 18 C 26 14, 30 14, 32 16 C 28 18, 24 19, 20 18 Z" fill={GREEN} />
          <path d="M15 26 C 10 22, 6 22, 4 24 C 8 26, 12 27, 15 26 Z" fill={GREEN} />
        </g>
      )
      break
    case 4: // 春 a sprout
      picture = (
        <g>
          <path d="M18 42 L18 22" stroke={GREEN} strokeWidth="2.2" />
          <path d="M18 28 C 10 28, 6 22, 6 16 C 12 16, 18 20, 18 28 Z" fill="#4caf50" />
          <path d="M18 24 C 26 24, 30 18, 30 12 C 24 12, 18 16, 18 24 Z" fill="#4caf50" />
          <circle cx="18" cy="42" r="0" />
        </g>
      )
      break
    case 5: // 夏 the sun
      picture = (
        <g stroke="#e07a00" strokeWidth="2" strokeLinecap="round">
          {Array.from({ length: 8 }, (_, i) => {
            const a = (i * 45 * Math.PI) / 180
            return <line key={i} x1={18 + 9 * Math.cos(a)} y1={26 + 9 * Math.sin(a)} x2={18 + 13 * Math.cos(a)} y2={26 + 13 * Math.sin(a)} />
          })}
          <circle cx="18" cy="26" r="7" fill="#f6b21a" stroke="none" />
        </g>
      )
      break
    case 6: // 秋 a falling leaf
      picture = (
        <g>
          <path d="M18 12 C 30 18, 30 32, 18 40 C 6 32, 6 18, 18 12 Z" fill="#d2691e" />
          <path d="M18 14 L18 44 M18 24 L24 20 M18 30 L12 26 M18 34 L24 30" stroke="#7a3a10" strokeWidth="1" fill="none" />
        </g>
      )
      break
    default: // 冬 a snowflake
      picture = (
        <g stroke="#3a8fd0" strokeWidth="1.8" strokeLinecap="round">
          {[0, 60, 120].map((d) => (
            <g key={d} transform={`rotate(${d} 18 26)`}>
              <line x1="18" y1="14" x2="18" y2="38" />
              <path d="M15 17 L18 20 L21 17 M15 35 L18 32 L21 35" fill="none" />
            </g>
          ))}
        </g>
      )
  }
  return (
    <g>
      <g transform="translate(6.5 9) scale(0.75)">{picture}</g>
      <text x="3" y="19" fontSize="20" fill={numberColour} fontWeight="700" fontFamily={SERIF}>
        {n}
      </text>
    </g>
  )
}

/** The face of tile `t` (index as in docs/rules/hk.md §1). */
export function TileFace({ t }: { t: number }) {
  if (t < 9) {
    return (
      <g fontFamily={SERIF} fontWeight="700" textAnchor="middle">
        <text x="18" y="22" fontSize="17" fill={INK}>
          {NUMERALS[t]}
        </text>
        <text x="18" y="42" fontSize="17" fill={RED}>
          萬
        </text>
      </g>
    )
  }
  if (t < 18) {
    const { r, spots } = DOTS[t - 9]
    return (
      <g>
        {spots.map(([x, y, c], i) => (
          <Dot key={i} x={x} y={y} r={r} colour={c} />
        ))}
      </g>
    )
  }
  if (t === 18) return <Bird />
  if (t < 27) {
    const { h, spots } = STICKS[t - 18]
    return (
      <g>
        {spots.map(([x, y, c, angle], i) => (
          <Stick key={i} x={x} y={y} h={h} colour={c} angle={angle} />
        ))}
      </g>
    )
  }
  if (t === 31) return <rect x="8" y="9" width="20" height="30" rx="2" fill="none" stroke={BLUE} strokeWidth="2.5" />
  if (t < 34) {
    const ch = ['東', '南', '西', '北', '白', '發', '中'][t - 27]
    const colour = t === 32 ? GREEN : t === 33 ? RED : '#222'
    return (
      <text x="18" y="33" fontSize="22" textAnchor="middle" fill={colour} fontFamily={SERIF} fontWeight="700">
        {ch}
      </text>
    )
  }
  return <Bonus k={t - 34} />
}
