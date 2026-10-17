import type { Meld, Observation } from '../api'
import { WINDS } from '../tiles'
import { Tile } from './Tile'

/** Screen position relative to the human: 0 bottom, 1 right, 2 top, 3 left. */
export type Position = 0 | 1 | 2 | 3

/** Side seats lie along the left and right edges, with their tiles on their side. */
function rotation(position: Position): 0 | 90 | -90 {
  return position === 1 ? -90 : position === 3 ? 90 : 0
}

export function MeldView({ meld, size = 'sm', rotate = 0 }: { meld: Meld; size?: 'sm' | 'md'; rotate?: 0 | 90 | -90 }) {
  return (
    <span className="meld">
      {meld.tiles.map((t, i) => (
        <Tile key={i} tile={t} size={size} rotate={rotate} faceDown={meld.concealed && (i === 0 || i === 3)} />
      ))}
    </span>
  )
}

interface EdgeProps {
  obs: Observation
  seat: number
  position: Position
  name: string
  revealed?: number[]
  delta?: number
}

/** A seat at the edge of the table: wind, name, concealed tiles, melds and
 * bonus tiles. Side seats run vertically. Discards are drawn separately, out on
 * the table ([`River`]). */
export function SeatEdge({ obs, seat, position, name, revealed, delta }: EdgeProps) {
  const rotate = rotation(position)
  const acting = obs.current_player === seat
  const hidden = position !== 0
  return (
    <div className={`edge pos-${position}${acting ? ' acting' : ''}`} data-testid={`seat-${seat}`}>
      <div className="seat-head">
        <span className="wind">{WINDS[seat]}</span>
        <span className="seat-name">{name}</span>
        {seat === 0 && <span className="badge">莊</span>}
        {obs.liable[seat] !== null && (
          <span className="badge warn" title="十二張落地：包自摸">
            包 {WINDS[obs.liable[seat]!]}
          </span>
        )}
        {delta !== undefined && (
          <span className={`delta ${delta >= 0 ? 'up' : 'down'}`}>
            {delta > 0 ? '+' : ''}
            {delta}
          </span>
        )}
      </div>
      <div className="edge-tiles">
        {hidden &&
          (revealed ? (
            <span className="tiles">{revealed.map((t, i) => <Tile key={i} tile={t} size="sm" rotate={rotate} />)}</span>
          ) : (
            <span className="tiles">
              {Array.from({ length: obs.concealed_counts[seat] }, (_, i) => (
                <Tile key={i} size="sm" faceDown rotate={rotate} />
              ))}
            </span>
          ))}
        {hidden && obs.melds[seat].map((m, i) => <MeldView key={i} meld={m} rotate={rotate} />)}
        {obs.bonus[seat].length > 0 && (
          <span className="tiles bonus">
            {obs.bonus[seat].map((t) => (
              <Tile key={t} tile={t} size="sm" rotate={rotate} />
            ))}
          </span>
        )}
      </div>
    </div>
  )
}

/** One seat's discards, laid out in the open between the seat and the centre.
 * The most recent discard is outlined; claimed ones are faded. */
export function River({ obs, seat, position }: { obs: Observation; seat: number; position: Position }) {
  const river = obs.river.filter(([s]) => s === seat)
  const last = obs.river.length > 0 ? obs.river[obs.river.length - 1] : null
  return (
    <div className={`river river-${position}`} data-testid={`river-${seat}`}>
      {river.map(([, t, claimed], i) => (
        <Tile
          key={i}
          tile={t}
          size="sm"
          dim={claimed}
          highlight={last !== null && i === river.length - 1 && last[0] === seat && !claimed}
        />
      ))}
    </div>
  )
}
