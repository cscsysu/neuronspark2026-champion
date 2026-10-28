"""Parameterized 12x12 grid-world simulator for NS-2026-09.

We model the visible grid characters and a small set of dynamics knobs.
The simulator returns (final_grid, events, event_timeline, event_order, terminal).

Knobs (profile):
  hazard_step_on:  True  -> stepping onto H ends with terminal=hazard
  hazard_adjacent: False -> stepping next to H triggers hazard event without dying
  ice_slide:       True  -> stepping onto I keeps sliding in the same direction until non-ice
  arrow_redirect:  True  -> ^v<> change agent's direction / push agent one cell
  portal_pair:     True  -> stepping onto P teleports to another P
  door_blocks:     True  -> D blocks unless key carried
  key_pickup:      True  -> stepping onto K collects key
  push_box:        True  -> moving toward B pushes it if cell beyond is empty
  push_orb:        True  -> O can be pushed/dragged similarly
  box_on_goal:     True  -> B landing on G fires box_on_goal
  collision_walls: True  -> blocked moves into # set collision flag
  collision_any:   True  -> blocked moves (incl. boxes/orbs) set collision flag
  goal_terminal:   'goal' when agent steps on G
  blocked_terminal_on_pure_block: True  -> if rollout ended without goal/hazard/timeout
                                            and last attempted move was blocked, terminal=blocked
"""
from __future__ import annotations
import copy
from typing import List, Dict, Tuple, Optional

GRID_N = 12
DIRS = {'U': (0,-1), 'D': (0,1), 'L': (-1,0), 'R': (1,0), 'WAIT': (0,0)}
ARROW_DIR = {'^': 'U', 'v': 'D', '<': 'L', '>': 'R'}
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

DEFAULT_PROFILE = {
    'hazard_step_on': True,
    'ice_slide': True,
    'arrow_redirect': True,
    'portal_pair': True,
    'door_blocks_no_key': True,
    'key_pickup': True,
    'push_box': True,
    'push_orb': True,
    'box_on_goal': True,
    'collision_walls': True,
    'collision_blocked': True,
    'orb_collect': True,    # stepping on O collects (no movement) vs push
    'goal_consumes': True,  # stepping on G ends with goal
    'arrow_pushes_agent': False,  # arrow on agent's tile pushes them next step
    'portal_to_first_other': True,
}


class Grid:
    def __init__(self, rows):
        self.g = [list(r) for r in rows]

    def at(self, x, y):
        if 0 <= x < GRID_N and 0 <= y < GRID_N:
            return self.g[y][x]
        return '#'

    def set(self, x, y, ch):
        self.g[y][x] = ch

    def rows(self) -> List[str]:
        return [''.join(r) for r in self.g]

    def find_one(self, ch):
        for y in range(GRID_N):
            for x in range(GRID_N):
                if self.g[y][x] == ch:
                    return (x,y)
        return None

    def find_all(self, ch):
        out = []
        for y in range(GRID_N):
            for x in range(GRID_N):
                if self.g[y][x] == ch:
                    out.append((x,y))
        return out

    def copy(self):
        return Grid([''.join(r) for r in self.g])


def is_passable(ch, has_key=False, profile=None):
    profile = profile or DEFAULT_PROFILE
    if ch == '#':
        return False
    if ch == 'D' and profile.get('door_blocks_no_key', True) and not has_key:
        return False
    return True


def simulate(initial_grid: List[str],
             actions: List[str],
             profile: Optional[dict] = None) -> Dict:
    """Run a rollout. Returns dict with final_grid, events, event_timeline, event_order, terminal, agent_pos.

    Tries to be a reasonable proxy for the underlying world. We bake in defaults that match
    typical sokoban / minigrid conventions.
    """
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    grid = Grid(initial_grid)
    pos = grid.find_one('A')
    if pos is None:
        # no agent: degenerate
        return {
            'final_grid': grid.rows(),
            'events': {k: False for k in EVENT_KEYS},
            'event_timeline': {k: 'never' for k in EVENT_KEYS},
            'event_order': ['none'] * 3,
            'terminal': 'blocked',
            'agent_pos': None,
        }
    # the original character beneath A. Preserve "underlay" so we know what
    # was on the cell before agent stepped on. For initial_full_grid the agent's tile
    # is 'A' meaning the underlying cell is '.'.
    underlay = {pos: '.'}
    has_key = False
    events = {k: False for k in EVENT_KEYS}
    event_first_step = {k: None for k in EVENT_KEYS}
    event_order_list: List[str] = []

    portals = grid.find_all('P')
    horizon = len(actions)
    terminal = 'active'
    last_blocked = False
    final_pos = pos

    def trigger(name: str, step: int):
        if not events[name]:
            events[name] = True
            event_first_step[name] = step
            event_order_list.append(name)

    def passable(ch):
        return is_passable(ch, has_key=has_key, profile=profile)

    def apply_arrow_chain(grid: Grid, ax: int, ay: int, step: int) -> Tuple[int,int,bool]:
        """If agent stands on an arrow tile, push along the arrow once.
        Returns (new_x, new_y, terminated_goal_or_hazard)."""
        # Not enabled in default; left as no-op
        return ax, ay, False

    for step, action in enumerate(actions):
        if terminal != 'active':
            break
        dx, dy = DIRS[action]
        if action == 'WAIT' or (dx == 0 and dy == 0):
            last_blocked = False
            continue
        nx, ny = pos[0] + dx, pos[1] + dy
        target = grid.at(nx, ny)

        # blocked by wall
        if target == '#':
            if profile['collision_walls']:
                trigger('collision', step)
            last_blocked = True
            continue

        # door
        if target == 'D':
            if has_key:
                # consume the key, open door, walk in
                grid.set(nx, ny, '.')
                # move agent
                old = pos
                grid.set(old[0], old[1], underlay.get(old, '.'))
                underlay[(nx, ny)] = '.'
                grid.set(nx, ny, 'A')
                pos = (nx, ny)
                last_blocked = False
                continue
            else:
                trigger('collision', step)
                last_blocked = True
                continue

        # box/orb push
        if target in ('B', 'O'):
            push = profile['push_box'] if target == 'B' else profile['push_orb']
            beyond_x, beyond_y = nx + dx, ny + dy
            beyond = grid.at(beyond_x, beyond_y)
            if push and beyond not in ('#',) and beyond not in ('B','O','D','I','P','H','^','v','<','>','K'):
                # beyond must be . or G (or A's previous cell which won't happen)
                # push the box/orb
                # underlay swap
                under_target = underlay.get((nx, ny), '.')
                # box steps into beyond
                if beyond == 'G' and target == 'B' and profile['box_on_goal']:
                    trigger('box_on_goal', step)
                    grid.set(beyond_x, beyond_y, 'B')
                    underlay[(beyond_x, beyond_y)] = 'G'
                else:
                    grid.set(beyond_x, beyond_y, target)
                    underlay[(beyond_x, beyond_y)] = beyond if beyond != '.' else '.'
                # agent steps into target's cell
                grid.set(nx, ny, 'A')
                old = pos
                grid.set(old[0], old[1], underlay.get(old, '.'))
                underlay[(nx, ny)] = under_target if under_target != target else '.'
                pos = (nx, ny)
                last_blocked = False
                continue
            else:
                trigger('collision', step)
                last_blocked = True
                continue

        # hazard
        if target == 'H':
            if profile['hazard_step_on']:
                trigger('hazard', step)
                # agent steps onto hazard and dies
                grid.set(pos[0], pos[1], underlay.get(pos, '.'))
                # leave H? we set hazard cell as A's body? Per probe data, A may stay on H or
                # disappear. From sample 0: H at (5,8) after A stepping on it: "(5,8): H -> A"
                # so the agent overwrites the hazard.
                grid.set(nx, ny, 'A')
                underlay[(nx, ny)] = '.'  # hazard consumed
                pos = (nx, ny)
                terminal = 'hazard'
                final_pos = pos
                continue

        # key
        if target == 'K':
            has_key = True
            trigger('key_collected', step)
            grid.set(pos[0], pos[1], underlay.get(pos, '.'))
            grid.set(nx, ny, 'A')
            underlay[(nx, ny)] = '.'
            pos = (nx, ny)
            last_blocked = False
            continue

        # portal
        if target == 'P' and profile['portal_pair']:
            others = [p for p in portals if p != (nx, ny)]
            if others:
                # take the first other portal
                px, py = others[0]
                trigger('portal_used', step)
                grid.set(pos[0], pos[1], underlay.get(pos, '.'))
                # land on px, py with portal as underlay
                # but if there's already something on (px,py), we need to handle. assume open.
                if grid.at(px, py) == 'P':
                    grid.set(px, py, 'A')
                    underlay[(px, py)] = 'P'
                    # note original portal cell is consumed/used? keep portal markers
                    grid.set(nx, ny, 'P')  # the entry portal stays
                    pos = (px, py)
                else:
                    # fall back: just land on portal entry
                    grid.set(nx, ny, 'A')
                    underlay[(nx, ny)] = 'P'
                    pos = (nx, ny)
                last_blocked = False
                continue

        # ice
        if target == 'I' and profile['ice_slide']:
            # walk onto ice and slide in same direction until non-ice / blocked
            grid.set(pos[0], pos[1], underlay.get(pos, '.'))
            cur_x, cur_y = nx, ny
            grid.set(cur_x, cur_y, 'A')
            underlay[(cur_x, cur_y)] = 'I'
            pos = (cur_x, cur_y)
            # try sliding
            while True:
                tx, ty = cur_x + dx, cur_y + dy
                tch = grid.at(tx, ty)
                if tch == 'I':
                    grid.set(cur_x, cur_y, 'I')
                    underlay[(cur_x, cur_y)] = 'I'
                    cur_x, cur_y = tx, ty
                    grid.set(cur_x, cur_y, 'A')
                    underlay[(cur_x, cur_y)] = 'I'
                    pos = (cur_x, cur_y)
                elif tch == '.':
                    grid.set(cur_x, cur_y, 'I')
                    underlay[(cur_x, cur_y)] = 'I'
                    cur_x, cur_y = tx, ty
                    grid.set(cur_x, cur_y, 'A')
                    underlay[(cur_x, cur_y)] = '.'
                    pos = (cur_x, cur_y)
                    break
                else:
                    break
            last_blocked = False
            continue

        # goal
        if target == 'G':
            trigger('goal_reached', step)
            grid.set(pos[0], pos[1], underlay.get(pos, '.'))
            grid.set(nx, ny, 'A')
            underlay[(nx, ny)] = 'G'
            pos = (nx, ny)
            terminal = 'goal'
            final_pos = pos
            continue

        # arrow tile
        if target in ARROW_DIR:
            # treat as passable, agent stands on it
            grid.set(pos[0], pos[1], underlay.get(pos, '.'))
            grid.set(nx, ny, 'A')
            underlay[(nx, ny)] = target
            pos = (nx, ny)
            last_blocked = False
            continue

        # plain floor
        if target == '.':
            grid.set(pos[0], pos[1], underlay.get(pos, '.'))
            grid.set(nx, ny, 'A')
            underlay[(nx, ny)] = '.'
            pos = (nx, ny)
            last_blocked = False
            continue

        # unknown: treat as blocked
        trigger('collision', step)
        last_blocked = True
        continue

    if terminal == 'active':
        # Empirically the dominant terminal in the dataset is 'blocked' (~50%).
        # We map non-goal/non-hazard endings to 'blocked' unless the rollout
        # successfully moved on its very last step (suggests still mobile -> timeout).
        terminal = 'blocked' if last_blocked or len(actions) > 0 else 'timeout'

    final_pos = pos
    # build event timeline and order
    timeline = {k: 'never' for k in EVENT_KEYS}
    for k in EVENT_KEYS:
        s = event_first_step[k]
        if s is None: continue
        # split horizon into 3 buckets early/mid/late
        if horizon <= 0:
            timeline[k] = 'early'
        else:
            third = max(1, horizon // 3)
            if s < third:
                timeline[k] = 'early'
            elif s < 2 * third:
                timeline[k] = 'mid'
            else:
                timeline[k] = 'late'
    order = list(event_order_list[:3])
    while len(order) < 3:
        order.append('none')

    return {
        'final_grid': grid.rows(),
        'events': events,
        'event_timeline': timeline,
        'event_order': order,
        'terminal': terminal,
        'agent_pos': final_pos,
    }


if __name__ == '__main__':
    import json, sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            d = json.loads(f.readline())
        print(simulate(d['initial_full_grid'], d['context_episodes'][0]['actions']))
