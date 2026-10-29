// Brian's 400-hand benchmark (plan Task 12): labels for the 基準測試 button.
import type { BenchmarkProgress, GameView } from './api'

export const BENCHMARK_TAG = 'benchmark'

export function isBenchmark(view: Pick<GameView, 'tag'> | null): boolean {
  return view?.tag === BENCHMARK_TAG
}

/** The button: `基準測試 12 / 400`; `續打` while a hand is under way; done at 400. */
export function benchmarkButtonLabel(p: BenchmarkProgress | null): string {
  if (!p) return '基準測試'
  if (p.next === null) return `基準測試完成 ${p.played} / ${p.total}`
  const resume = p.in_progress !== null ? '（續打）' : ''
  return `基準測試 ${p.played} / ${p.total}${resume}`
}

/** The header during a benchmark hand: which hand of the 400 this is (1-based). */
export function benchmarkHeader(index: number, total = 400): string {
  return `基準測試 第 ${index + 1} / ${total} 局`
}
