import { expect, test } from '@playwright/test'

// Plays a whole hand in the browser against the easy bots, with study hints on,
// always taking the first offered action (wins first) or discarding the first
// tile. Checks that the hand ends, the result is shown, and the review loads.
test('play a hand with hints and review it', async ({ page }) => {
  await page.goto('/?botDelay=0&hintMs=150')
  await page.getByRole('radio', { name: '初級' }).check()
  await page.getByRole('radio', { name: /^東/ }).check()
  await page.getByRole('button', { name: '開始' }).click()
  await page.getByRole('button', { name: '提示' }).click()

  // The dealer moves first: hints appear for the opening discard.
  await expect(page.getByTestId('hint-moves')).toBeVisible({ timeout: 30_000 })

  for (let step = 0; step < 400; step++) {
    if (await page.getByTestId('game-over').isVisible()) break
    const prompt = page.getByTestId('prompt')
    if (!(await prompt.isVisible())) {
      await page.waitForTimeout(50)
      continue
    }
    const win = page.locator('.act-104, .act-34')
    if (await win.count()) {
      await win.first().click()
      continue
    }
    const pass = page.locator('.act-103')
    if (await pass.count()) {
      await pass.click()
      continue
    }
    // Discard: select a tile, then press 打出.
    const tile = page.locator('[data-testid="hand"] .tile.clickable').last()
    if (!(await tile.count())) continue
    await tile.click({ force: true })
    const discard = page.getByRole('button', { name: '打出' })
    if (await discard.isEnabled()) await discard.click()
  }
  await expect(page.getByTestId('game-over')).toBeVisible({ timeout: 60_000 })
  await page.getByRole('button', { name: '檢討' }).click()
  await expect(page.getByTestId('review')).toBeVisible({ timeout: 120_000 })
})
