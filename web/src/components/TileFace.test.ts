import { describe, expect, it } from 'vitest'
import { DOTS, STICKS } from './TileFace'

describe('tile faces', () => {
  it('draws n dots and n sticks on the n tile', () => {
    DOTS.forEach((d, i) => expect(d.spots.length).toBe(i + 1))
    STICKS.slice(1).forEach((s, i) => expect(s.spots.length).toBe(i + 2))
  })
  it('keeps every spot on the face', () => {
    for (const { spots } of [...DOTS, ...STICKS]) {
      for (const [x, y] of spots) {
        expect(x).toBeGreaterThan(3)
        expect(x).toBeLessThan(33)
        expect(y).toBeGreaterThan(3)
        expect(y).toBeLessThan(45)
      }
    }
  })
})
