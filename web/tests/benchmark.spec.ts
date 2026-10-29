import { expect, test } from '@playwright/test'

// Plays one benchmark hand (plan Task 12) in the browser: the 基準測試 button shows
// "n / 400", the hand has no study panel and no review, and the count goes up by one.
// The server logs these hands under web/test-results/ (playwright.config.ts), never
// in Brian's runs/benchmark/hands.
test('play one benchmark hand', async ({ page }) => {
  await page.goto('/?botDelay=0')
  const start = page.getByTestId('benchmark-start')
  await expect(start).toHaveText(/^基準測試 \d+ \/ 400/)
  const played = Number(/(\d+) \/ 400/.exec((await start.textContent()) ?? '')![1])
  await start.click()

  await expect(page.getByTestId('benchmark-header')).toBeVisible()
  await expect(page.locator('.study')).toHaveCount(0)
  await page.screenshot({ path: 'test-results/benchmark-hand.png' })

  const quick = { timeout: 3000 }
  for (let step = 0; step < 600; step++) {
    if (await page.getByTestId('game-over').isVisible()) break
    if (!(await page.getByTestId('prompt').isVisible())) {
      await page.waitForTimeout(50)
      continue
    }
    try {
      const win = page.locator('.act-104, .act-34')
      if (await win.count()) {
        await win.first().click(quick)
        continue
      }
      const pass = page.locator('.act-103')
      if (await pass.count()) {
        await pass.click(quick)
        continue
      }
      const tile = page.locator('[data-testid="hand"] .tile.clickable').last()
      await tile.click({ ...quick, force: true })
      await page.getByRole('button', { name: '打出' }).click(quick)
    } catch {
      // The view changed under us (a bot moved); look again.
    }
  }
  await expect(page.getByTestId('game-over')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('button', { name: '檢討' })).toHaveCount(0)
  await expect(page.getByTestId('benchmark-next')).toContainText(`${played + 1} / 400`)
  await page.screenshot({ path: 'test-results/benchmark-over.png' })
})
