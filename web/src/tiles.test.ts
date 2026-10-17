import { describe, expect, it } from 'vitest'
import { faceText, planName, relativeSeat, sortTiles, suitOf, tileKind, tileName } from './tiles'

describe('tiles', () => {
  it('names every tile in Chinese', () => {
    expect(tileName(0)).toBe('一萬')
    expect(tileName(13)).toBe('五筒')
    expect(tileName(26)).toBe('九索')
    expect(tileName(27)).toBe('東')
    expect(tileName(31)).toBe('白')
    expect(tileName(33)).toBe('中')
    expect(tileName(35)).toBe('蘭')
    expect(tileName(41)).toBe('冬')
  })

  it('classifies tiles', () => {
    expect(tileKind(5)).toBe('suit')
    expect(tileKind(29)).toBe('wind')
    expect(tileKind(32)).toBe('dragon')
    expect(tileKind(38)).toBe('bonus')
    expect(suitOf(20)).toBe(2)
    expect(suitOf(30)).toBe(-1)
  })

  it('draws faces', () => {
    expect(faceText(8)).toEqual({ top: '九', bottom: '萬' })
    expect(faceText(32)).toEqual({ top: '發', bottom: '' })
    expect(faceText(39)).toEqual({ top: '夏', bottom: '2' })
  })

  it('sorts and places seats', () => {
    expect(sortTiles([33, 2, 9])).toEqual([2, 9, 33])
    expect(relativeSeat(1, 0)).toBe(1)
    expect(relativeSeat(0, 1)).toBe(3)
  })

  it('names plans', () => {
    expect(planName({ plan: 'half_flush_p', structure: 'all_pungs', pungs: [33], limit: false })).toBe(
      '混一色（筒） · 對對糊 · 中',
    )
    expect(planName({ plan: 'mixed', structure: 'any', pungs: [], limit: false })).toBe('門前清及花')
  })
})
