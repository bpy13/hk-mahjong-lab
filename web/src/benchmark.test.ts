import { describe, expect, it } from 'vitest'
import { benchmarkButtonLabel, benchmarkHeader, isBenchmark } from './benchmark'

const progress = { played: 12, total: 400, next: 12, in_progress: null, opponents: 'value' }

describe('benchmark', () => {
  it('labels the button with the hands played', () => {
    expect(benchmarkButtonLabel(null)).toBe('基準測試')
    expect(benchmarkButtonLabel(progress)).toBe('基準測試 12 / 400')
    expect(benchmarkButtonLabel({ ...progress, in_progress: 12 })).toBe('基準測試 12 / 400（續打）')
    expect(benchmarkButtonLabel({ ...progress, played: 400, next: null })).toBe('基準測試完成 400 / 400')
  })

  it('names the hand and spots benchmark views', () => {
    expect(benchmarkHeader(0)).toBe('基準測試 第 1 / 400 局')
    expect(isBenchmark({ tag: 'benchmark' })).toBe(true)
    expect(isBenchmark({ tag: null })).toBe(false)
    expect(isBenchmark(null)).toBe(false)
  })
})
