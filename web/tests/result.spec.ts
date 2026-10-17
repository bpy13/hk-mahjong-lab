import { expect, test } from '@playwright/test'
// Plays a hand, then checks the result window closes (button, Esc) and reopens.
test('the result window can be closed and reopened', async ({ page }) => {
  await page.goto('/?botDelay=0')
  await page.getByRole('radio', { name: '初級' }).check()
  await page.getByRole('radio', { name: /^南/ }).check()
  await page.getByRole('button', { name: '開始' }).click()
  const quick = { timeout: 3000 }
  for (let step = 0; step < 600; step++) {
    if (await page.getByTestId('game-over').isVisible()) break
    if (!(await page.getByTestId('prompt').isVisible())) { await page.waitForTimeout(50); continue }
    try {
      const pass = page.locator('.act-103')
      if (await pass.count()) { await pass.click(quick); continue }
      const tile = page.locator('[data-testid="hand"] .tile.clickable').last()
      await tile.click({ ...quick, force: true })
      await page.getByRole('button', { name: '打出' }).click(quick)
    } catch { /* view changed */ }
  }
  await expect(page.getByTestId('game-over')).toBeVisible({ timeout: 60000 })
  await page.getByRole('button', { name: '關閉' }).first().click()
  await expect(page.getByTestId('game-over')).toBeHidden()
  await page.getByRole('button', { name: '結果' }).click()
  await expect(page.getByTestId('game-over')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('game-over')).toBeHidden()
})
