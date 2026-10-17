import { Tile } from './Tile'

/** The tiles left to draw, face down, with their count beneath. */
export function Wall({ left }: { left: number }) {
  return (
    <div className="wall" title={`餘牌 ${left} 隻`}>
      <div className="wall-tiles">
        {Array.from({ length: left }, (_, i) => (
          <Tile key={i} size="xs" faceDown />
        ))}
      </div>
      <span className="wall-count">
        餘牌 <span data-testid="wall">{left}</span>
      </span>
    </div>
  )
}
