import { isDiscard } from '../tiles'

interface Props {
  legal: { action: number; label: string }[]
  selected: number | null
  onAct: (a: number) => void
  busy: boolean
}

/** Buttons for every legal action that is not a plain discard, plus 打出 for the
 * selected tile. */
export function ActionBar({ legal, selected, onAct, busy }: Props) {
  const special = legal.filter((l) => !isDiscard(l.action))
  const canDiscard = selected !== null && legal.some((l) => l.action === selected)
  return (
    <div className="action-bar">
      {special.map((l) => (
        <button key={l.action} className={`act act-${l.action}`} disabled={busy} onClick={() => onAct(l.action)}>
          {l.label}
        </button>
      ))}
      {legal.some((l) => isDiscard(l.action)) && (
        <button className="act discard" disabled={busy || !canDiscard} onClick={() => selected !== null && onAct(selected)}>
          打出
        </button>
      )}
    </div>
  )
}
