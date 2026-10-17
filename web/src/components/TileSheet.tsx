import { Tile } from './Tile'
import type { TileSize } from './Tile'
import { tileName } from '../tiles'

const GROUPS: [string, number[]][] = [
  ['萬子', [0, 1, 2, 3, 4, 5, 6, 7, 8]],
  ['筒子', [9, 10, 11, 12, 13, 14, 15, 16, 17]],
  ['索子', [18, 19, 20, 21, 22, 23, 24, 25, 26]],
  ['風牌', [27, 28, 29, 30]],
  ['三元牌', [31, 32, 33]],
  ['花', [34, 35, 36, 37]],
  ['季', [38, 39, 40, 41]],
]

/** Every tile face at several sizes, for reviewing the artwork (`#tiles`). */
export function TileSheet() {
  return (
    <div className="sheet">
      <h1>牌樣</h1>
      {GROUPS.map(([name, tiles]) => (
        <section key={name}>
          <h2>{name}</h2>
          <div className="sheet-row">
            {tiles.map((t) => (
              <figure key={t}>
                <Tile tile={t} size="xl" />
                <figcaption>{tileName(t)}</figcaption>
              </figure>
            ))}
          </div>
          <div className="sheet-row small">
            {(['md', 'sm'] as TileSize[]).map((size) => tiles.map((t) => <Tile key={`${size}${t}`} tile={t} size={size} />))}
          </div>
          <div className="sheet-row small">
            {tiles.map((t) => (
              <Tile key={`r${t}`} tile={t} size="sm" rotate={90} />
            ))}
            {tiles.map((t) => (
              <Tile key={`l${t}`} tile={t} size="sm" rotate={-90} />
            ))}
          </div>
        </section>
      ))}
      <section>
        <h2>牌背</h2>
        <div className="sheet-row">
          <Tile size="xl" faceDown />
          <Tile size="sm" faceDown />
          <Tile size="xs" faceDown />
        </div>
      </section>
    </div>
  )
}
