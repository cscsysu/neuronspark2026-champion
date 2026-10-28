"""Simulator v2: G and H trigger events but DO NOT end the rollout.

Key changes:
- Agent stepping on G fires goal_reached but continues moving (G stays in place).
- Agent stepping on H fires hazard but continues moving (H stays in place).
- Final terminal computed AFTER all actions:
  - 'goal' if goal_reached event fired (highest priority)
  - 'hazard' if hazard event fired
  - 'blocked' if last action was blocked
  - 'timeout' otherwise

This matches train data: 89% of terminal=goal probes don't end at G.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

GRID_N = 12
DIRS = {'U': (0,-1), 'D': (0,1), 'L': (-1,0), 'R': (1,0), 'WAIT': (0,0)}
ARROW_DIR = {'^': 'U', 'v': 'D', '<': 'L', '>': 'R'}
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

DEFAULT_PROFILE = {
    'hazard_passable': True,
    'goal_passable': True,
    'ice_slide': True,
    'ice_mode': 'classic',  # 'classic', 'no_slide', 'slide_through', 'slide_one'
    'arrow_redirect': False,
    'arrow_mode': 'pass',  # 'pass', 'redirect', 'block'
    'portal_pair': True,
    'door_blocks_no_key': True,
    'door_no_block': False,
    'key_pickup': True,
    'push_box': True,
    'box_walk': False,
    'push_orb': True,
    'orb_blocks': False,
    'hazard_kills': False,
    'goal_kills': False,
}


def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x, y)
    return None


def find_all(grid, ch):
    out = []
    for y, r in enumerate(grid):
        for x, c in enumerate(r):
            if c == ch: out.append((x, y))
    return out


def simulate(initial_grid: List[str],
             actions: List[str],
             profile: Optional[dict] = None) -> Dict:
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    g = [list(r) for r in initial_grid]
    pos = find_pos(initial_grid, 'A')
    if pos is None:
        return {
            'final_grid': [''.join(r) for r in g],
            'events': {k: False for k in EVENT_KEYS},
            'event_timeline': {k: 'never' for k in EVENT_KEYS},
            'event_order': ['none'] * 3,
            'terminal': 'blocked',
            'agent_pos': None,
        }
    # underlay: what was beneath agent's tile (start: '.')
    underlay = '.'
    has_key = False
    portals = find_all(initial_grid, 'P')
    events = {k: False for k in EVENT_KEYS}
    first_step = {k: None for k in EVENT_KEYS}
    order_list: List[str] = []
    horizon = len(actions)
    last_blocked = False

    def trigger(name: str, step: int):
        if not events[name]:
            events[name] = True
            first_step[name] = step
            order_list.append(name)

    def is_block(ch):
        if ch == '#': return True
        if ch == 'D' and not (profile['door_blocks_no_key'] and has_key):
            # door blocks unless we have key
            return profile['door_blocks_no_key']
        return False

    # Clear agent tile in working grid
    g[pos[1]][pos[0]] = underlay

    for step, action in enumerate(actions):
        if action == 'WAIT' or action not in DIRS:
            last_blocked = False
            continue
        dx, dy = DIRS[action]
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < GRID_N and 0 <= ny < GRID_N):
            trigger('collision', step)
            last_blocked = True
            continue
        target = g[ny][nx]
        # Wall
        if target == '#':
            trigger('collision', step)
            last_blocked = True
            continue
        # Door
        if target == 'D':
            if profile.get('door_no_block'):
                pos = (nx, ny)
                underlay = '.'
                last_blocked = False
                continue
            if profile['door_blocks_no_key'] and has_key:
                # consume key, walk in
                has_key = False
                g[ny][nx] = '.'
                pos = (nx, ny)
                underlay = '.'
                last_blocked = False
                continue
            else:
                trigger('collision', step)
                last_blocked = True
                continue
        # Box: walk_through, push, or block
        if target == 'B':
            if profile.get('box_walk'):
                # Agent walks over (box self-moves; sim removes from grid for tracking)
                g[ny][nx] = '.'
                pos = (nx, ny)
                underlay = '.'
                last_blocked = False
                continue
            if profile['push_box']:
                bx, by = nx + dx, ny + dy
                if 0 <= bx < GRID_N and 0 <= by < GRID_N and g[by][bx] in '.G':
                    if g[by][bx] == 'G':
                        trigger('box_on_goal', step)
                    g[by][bx] = 'B'
                    g[ny][nx] = '.'
                    pos = (nx, ny)
                    underlay = '.'
                    last_blocked = False
                    continue
            trigger('collision', step)
            last_blocked = True
            continue
        # Orb: walk over (default), or block
        if target == 'O':
            if profile.get('orb_blocks'):
                trigger('collision', step)
                last_blocked = True
                continue
            g[ny][nx] = '.'
            pos = (nx, ny)
            underlay = '.'
            last_blocked = False
            continue
        # Hazard: PASSABLE (don't end rollout) by default; optional kills
        if target == 'H':
            trigger('hazard', step)
            pos = (nx, ny)
            underlay = 'H'
            last_blocked = False
            if profile.get('hazard_kills'):
                break
            continue
        # Goal: PASSABLE by default
        if target == 'G':
            trigger('goal_reached', step)
            pos = (nx, ny)
            underlay = 'G'
            last_blocked = False
            if profile.get('goal_kills'):
                break
            continue
        # Key: pickup
        if target == 'K':
            if profile['key_pickup']:
                has_key = True
                trigger('key_collected', step)
                g[ny][nx] = '.'
                pos = (nx, ny)
                underlay = '.'
                last_blocked = False
                continue
        # Portal
        if target == 'P' and profile['portal_pair']:
            others = [p for p in portals if p != (nx, ny)]
            if others:
                trigger('portal_used', step)
                px, py = others[0]
                # land at other portal
                pos = (px, py)
                underlay = 'P'
                last_blocked = False
                continue
            # no other portal, just walk on it
            pos = (nx, ny)
            underlay = 'P'
            last_blocked = False
            continue
        # Ice
        if target == 'I':
            ice_mode = profile.get('ice_mode', 'classic' if profile.get('ice_slide', True) else 'no_slide')
            if ice_mode == 'classic':
                cur_x, cur_y = nx, ny
                while True:
                    tx, ty = cur_x + dx, cur_y + dy
                    if not (0 <= tx < GRID_N and 0 <= ty < GRID_N):
                        break
                    tch = g[ty][tx]
                    if tch == 'I':
                        cur_x, cur_y = tx, ty
                    elif tch == '.':
                        cur_x, cur_y = tx, ty
                        break
                    else:
                        break
                pos = (cur_x, cur_y)
                tile_under = g[cur_y][cur_x]
                underlay = tile_under if tile_under != '.' else '.'
                last_blocked = False
                continue
            elif ice_mode == 'slide_through':
                cur_x, cur_y = nx, ny
                while True:
                    tx, ty = cur_x + dx, cur_y + dy
                    if not (0 <= tx < GRID_N and 0 <= ty < GRID_N): break
                    if g[ty][tx] == '#': break
                    cur_x, cur_y = tx, ty
                pos = (cur_x, cur_y)
                tile_under = g[cur_y][cur_x]
                underlay = tile_under if tile_under != '.' else '.'
                last_blocked = False
                continue
            elif ice_mode == 'slide_one':
                tx, ty = nx + dx, ny + dy
                if 0 <= tx < GRID_N and 0 <= ty < GRID_N and g[ty][tx] != '#':
                    pos = (tx, ty)
                    tile_under = g[ty][tx]
                    underlay = tile_under if tile_under != '.' else '.'
                else:
                    pos = (nx, ny)
                    underlay = 'I'
                last_blocked = False
                continue
            else:  # no_slide
                pos = (nx, ny)
                underlay = 'I'
                last_blocked = False
                continue
        # Arrow (passable, optionally redirect)
        if target in ARROW_DIR:
            arrow_mode = profile.get('arrow_mode', 'pass')
            pos = (nx, ny)
            underlay = target
            if arrow_mode == 'redirect':
                adx, ady = DIRS[ARROW_DIR[target]]
                tx, ty = nx + adx, ny + ady
                if 0 <= tx < GRID_N and 0 <= ty < GRID_N and g[ty][tx] != '#':
                    pos = (tx, ty)
                    underlay = g[ty][tx] if g[ty][tx] != '.' else '.'
            last_blocked = False
            continue
        # Plain
        if target == '.':
            pos = (nx, ny)
            underlay = '.'
            last_blocked = False
            continue
        # Unknown -> treat as walked over
        pos = (nx, ny)
        underlay = target
        last_blocked = False

    # Determine terminal
    if events['goal_reached']:
        terminal = 'goal'
    elif events['hazard']:
        terminal = 'hazard'
    elif last_blocked:
        terminal = 'blocked'
    else:
        # finished without blocks and without goal/hazard => could be timeout or blocked
        terminal = 'blocked'  # blocked is most common in dataset

    # Place agent on grid
    final_g = [r[:] for r in g]
    if pos is not None:
        ax, ay = pos
        # If agent's current cell already has something static, place A on top
        final_g[ay][ax] = 'A'

    # Build event timeline
    timeline = {k: 'never' for k in EVENT_KEYS}
    if horizon > 0:
        for k in EVENT_KEYS:
            s = first_step[k]
            if s is None: continue
            third = max(1, horizon // 3)
            if s < third: timeline[k] = 'early'
            elif s < 2 * third: timeline[k] = 'mid'
            else: timeline[k] = 'late'
    order = list(order_list[:3])
    while len(order) < 3: order.append('none')

    return {
        'final_grid': [''.join(r) for r in final_g],
        'events': events,
        'event_timeline': timeline,
        'event_order': order,
        'terminal': terminal,
        'agent_pos': pos,
        'first_step': first_step,
        'horizon': horizon,
        'last_blocked': last_blocked,
    }
