import { sortTiles } from '../tiles'
import { Tile } from './Tile'

interface Props {
  hand: number[]
  lastDrawn: number | null
  canDiscard: boolean
  selected: number | null
  onSelect: (t: number) => void
  danger?: number[]
}

/** The human's concealed tiles. The tile just drawn sits apart on the right.
 * Click a tile to select it, click it again (or press Enter) to discard it. */
export function Hand({ hand, lastDrawn, canDiscard, selected, onSelect, danger }: Props) {
  const tiles = sortTiles(hand)
  let drawn: number | null = null
  if (lastDrawn !== null) {
    const i = tiles.lastIndexOf(lastDrawn)
    if (i >= 0) {
      tiles.splice(i, 1)
      drawn = lastDrawn
    }
  }
  const tile = (t: number, key: string) => (
    <Tile
      key={key}
      tile={t}
      size="lg"
      selected={selected === t}
      danger={danger ? danger[t] : undefined}
      reserveDanger
      onClick={canDiscard ? () => onSelect(t) : undefined}
      testId={`hand-${t}`}
    />
  )
  return (
    <div className="hand" data-testid="hand">
      {tiles.map((t, i) => tile(t, `h${i}`))}
      {/* The drawn tile's slot is always there, so the hand never shifts. */}
      <span className="drawn">{drawn !== null ? tile(drawn, 'drawn') : <span className="drawn-slot" />}</span>
    </div>
  )
}
