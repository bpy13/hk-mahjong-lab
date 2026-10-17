# Web UI and study mode

## Running it

```sh
conda activate mahjong
make serve            # builds web/ and serves UI + API on http://127.0.0.1:8000
```

During UI development, run the API and the Vite dev server side by side (the dev
server forwards `/api` to port 8000 and reloads on save):

```sh
uvicorn mahjong.server.app:app --reload     # terminal 1
make web-dev                                # terminal 2 → http://127.0.0.1:5173
```

URL options: `?botDelay=0` makes bots move instantly (default 600 ms per move);
`?hintMs=500` changes the solver's thinking time for hints (default 2000 ms).

The UI is in Traditional Chinese (繁體中文).

## Playing

* You sit at the bottom; the next player (下家) is on your right. As at a real
  table, each player's concealed tiles and melds run along their edge (the side
  players' vertically), and their discards lie out on the felt in front of them,
  growing toward the centre. The latest discard is outlined; claimed ones fade.
* On your turn, click a tile to raise it and click it again (or press **Enter**) to
  discard it (打出). The tile you just drew sits apart on the right.
* When you can claim a discard, buttons appear: 食糊 (win), 碰, 槓, 上 (chow, one
  button per way), 過 (pass, also **P**). On your turn: 自摸, 暗槓, 加槓.
* **新局** chooses the opponents, your seat and every rule variant in
  [rules/hk.md](rules/hk.md).

| Level | Bot | About |
|---|---|---|
| 初級 Easy | `efficiency` | Plays for any complete hand as fast as possible and ignores the faan minimum, so it often cannot win |
| 中級 Medium | `defensive` | Plays for hands that meet the minimum; folds when far from ready against a threat |
| 高級 Hard | `pimc:samples=96,candidates=4,ms=500` | The solver, about 0.5 s per decision |

## Study mode (研究)

The panel on the right has three settings:

* **關**: no solver output.
* **提示 (hints)**: on each of your decisions, the solver's top 3 moves with
  expected points (期望分), win rate (勝率) and deal-in rate (出銃率). Click a row to
  select that tile (or take that action). Below it:
  * 向聽 (shanten) and 番數向聽 (faan-aware shanten), with the target the
    faan-aware measure is following and the faan it guarantees;
  * on your turn, a discard table: faan-aware shanten after each discard, its
    effective tiles (有效張), plain shanten and acceptance, and 自摸率, the
    offense-only chance of reaching a legal self-drawn win.
* **測驗 (quiz)**: the solver works out its answer silently. After you move, it
  says whether you matched its top move and, if not, the difference in expected
  points.

**顯示危險度** draws a bar under each tile in your hand: the heuristic danger
model's estimate that discarding it lets an opponent win (green safe, amber, red).

### How to read the numbers

* **期望分** is the average points of the simulated hands: each sample re-deals the
  hidden tiles (with opponents' hands as developed as real ones at this stage),
  then the `fast` bot plays all four seats to the end.
* **±** is a 95% interval from sampling noise only. It does not cover the model's
  assumptions (how opponents are simulated and sampled). See
  [solver.md](solver.md).
* The bots themselves act on the paired difference from the `value` heuristic, so
  the solver's first row is not always what the 高級 bot would play.

### End of the hand

A window shows every winner's hand, the faan breakdown and the payments. Close
it with ✕, 關閉, Esc or a click outside to look at the final table (all hands are
revealed); **結果** in the header reopens it.

### Post-hand review (檢討)

After a hand, **檢討** re-analyzes each of your decisions that had a choice (300
samples, 8 candidates) and lists those where your move scored more than half a
point below the solver's best, with the total.

## Game logs

Every finished hand is written to `runs/games/<id>.json` (seed, rules, level, your
seat, every action and the result), enough to replay it exactly:
`HkState(seed, **rules)` then apply the actions in order.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/meta` | Levels, faan-table names, tile names |
| POST | `/api/games` | New game: `{level, seat, seed, rules}` |
| GET | `/api/games/{id}` | Your view of the game |
| POST | `/api/games/{id}/act` | Your action: `{action}` (an action code) |
| POST | `/api/games/{id}/step` | Let the next bot move once |
| GET | `/api/games/{id}/hints?ms=` | Solver, danger, progress and discard table |
| GET | `/api/games/{id}/review` | Post-hand review |
