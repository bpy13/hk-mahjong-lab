import { tileName } from '../tiles'
import { TileFace } from './TileFace'

export type TileSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl'

const SIZES: Record<TileSize, [number, number]> = { xs: [13, 17], sm: [26, 35], md: [36, 48], lg: [46, 62], xl: [72, 96] }

interface Props {
  tile?: number
  size?: TileSize
  faceDown?: boolean
  selected?: boolean
  dim?: boolean
  highlight?: boolean
  /** 0–1: drawn as a coloured bar under the tile (full width at 15%). */
  danger?: number
  /** Turn the tile on its side: 90 = top to the right, -90 = top to the left. */
  rotate?: 0 | 90 | -90
  onClick?: () => void
  testId?: string
}

/** One tile, drawn as SVG so it scales cleanly. */
export function Tile({ tile, size = 'md', faceDown, selected, dim, highlight, danger, rotate = 0, onClick, testId }: Props) {
  const [w, h] = SIZES[size]
  const name = faceDown || tile === undefined ? '牌背' : tileName(tile)
  const label = danger !== undefined ? `${name}（危險度 ${(danger * 100).toFixed(1)}%）` : name
  // A sideways tile is drawn in a swapped box, so it takes up its real space.
  const turned = rotate !== 0
  const group = rotate === 90 ? 'translate(48,0) rotate(90)' : rotate === -90 ? 'translate(0,36) rotate(-90)' : undefined
  return (
    <span
      className={`tile${selected ? ' selected' : ''}${dim ? ' dim' : ''}${onClick ? ' clickable' : ''}`}
      onClick={onClick}
      title={label}
      aria-label={label}
      role={onClick ? 'button' : 'img'}
      data-testid={testId}
      data-tile={tile}
    >
      <svg
        style={{ width: `calc(${turned ? h : w}px * var(--s, 1))`, height: `calc(${turned ? w : h}px * var(--s, 1))` }}
        viewBox={turned ? '0 0 48 36' : '0 0 36 48'}>
        <g transform={group}>
        <rect
          x="1"
          y="1"
          width="34"
          height="46"
          rx="5"
          fill={faceDown ? '#2f7d5b' : '#fbf7ec'}
          stroke={highlight ? '#e0a100' : '#7a7468'}
          strokeWidth={highlight ? 3 : 1.5}
        />
        {!faceDown && tile !== undefined && <TileFace t={tile} />}
        </g>
      </svg>
      {danger !== undefined && (
        <span className="danger-bar" style={{ width: `${Math.round(Math.min(1, danger / 0.15) * 100)}%`, background: dangerColour(danger) }} />
      )}
    </span>
  )
}

/** Green (< 1%) → amber (< 4%) → red as the deal-in estimate grows. */
export function dangerColour(d: number): string {
  if (d < 0.01) return '#3c9a5f'
  if (d < 0.04) return '#d8a106'
  return '#c8412b'
}
