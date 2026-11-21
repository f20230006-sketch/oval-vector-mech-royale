"""Pilot for the public mechfight/1 competition protocol."""

import array
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from mechfight import run


TAU = math.tau
PICKUP_VALUES = {"gun": 0, "mortar": 1, "lightning": 3, "laser": 5}
UPGRADE_VALUES = {
    "laser_charge": 12, "laser_damage": 11, "plating": 10, "nanites": 9.5,
    "armor": 9, "servos": 8.5, "gun_mag": 8, "gun_damage": 7,
    "gun_rate": 6, "reactor": 6, "lightning_range": 6, "jets": 5.5,
    "lightning_efficiency": 5, "gyros": 4, "capacitor": 4,
    "gun_velocity": 3, "mortar_reload": 2, "sensors": 2,
}


def clamp(value, low, high):
    return max(low, min(high, value))


def angle(value):
    return (value + math.pi) % TAU - math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def inside(x, y, edges):
    result = False
    for ax, ay, bx, by in edges:
        if (ay > y) != (by > y) and x < ax + (bx - ax) * (y - ay) / (by - ay):
            result = not result
    return result


def polygon(raw):
    vertices = list(zip(raw[::2], raw[1::2]))
    return [a + b for a, b in zip(vertices[-1:] + vertices[:-1], vertices)]


class Arena:
    def __init__(self, root):
        root = Path(root) / "v2"
        if not root.is_dir() and Path("/arena/v2").is_dir():
            root = Path("/arena/v2")
        world = json.loads((root / "world.json").read_text(encoding="utf-8"))
        self.bounds = world["bounds"]
        terrain = world["terrain"]
        self.x0, self.y0 = terrain["minX"], terrain["minY"]
        self.width, self.height = terrain["nx"], terrain["ny"]
        self.terrain = array.array("h", (root / "terrain.i16").read_bytes())
        if sys.byteorder != "little":
            self.terrain.byteswap()
        self.bins = defaultdict(list)
        for solid in world["prisms"]:
            xs, ys = solid["ring"][::2], solid["ring"][1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            item = (x0, x1, y0, y1, solid["base"], solid["top"],
                    polygon(solid["ring"]), [polygon(hole) for hole in solid["holes"]])
            for ix in range(math.floor(x0 / 16), math.floor(x1 / 16) + 1):
                for iy in range(math.floor(y0 / 16), math.floor(y1 / 16) + 1):
                    self.bins[ix, iy].append(item)

    def ground(self, x, y):
        gx = clamp(x - self.x0, 0, self.width - 1.001)
        gy = clamp(y - self.y0, 0, self.height - 1.001)
        ix, iy = int(gx), int(gy)
        fx, fy = gx - ix, gy - iy
        i = iy * self.width + ix
        lo = self.terrain[i] * (1 - fx) + self.terrain[i + 1] * fx
        hi = self.terrain[i + self.width] * (1 - fx) + self.terrain[i + self.width + 1] * fx
        return (lo * (1 - fy) + hi * fy) * .01

    def solid(self, x, y, lo, hi):
        for x0, x1, y0, y1, base, top, ring, holes in self.bins.get(
            (math.floor(x / 16), math.floor(y / 16)), ()
        ):
            if (x0 <= x <= x1 and y0 <= y <= y1 and lo < top and hi > base
                    and inside(x, y, ring) and not any(inside(x, y, hole) for hole in holes)):
                return True
        return False

    def ray(self, origin, target, gravity=0, flight=0):
        dx, dy, dz = (b - a for a, b in zip(origin, target))
        count = max(2, int(math.hypot(dx, dy) / 3.0))
        arc = .5 * gravity * flight * flight
        for i in range(1, count):
            frac = i / count
            x, y = origin[0] + frac * dx, origin[1] + frac * dy
            z = origin[2] + frac * dz + arc * frac * (1 - frac)
            if self.solid(x, y, z - .2, z + .2) or self.ground(x, y) > z:
                return False
        return True

    def clearance(self, pos, ux, uy, radius, height, goal_distance):
        x, y, z = pos
        limit = min(23, max(5, goal_distance))
        for step in (4, 8, 14, 22):
            span = min(step, limit)
            px, py = x + ux * span, y + uy * span
            if not (self.bounds["minX"] + radius < px < self.bounds["maxX"] - radius
                    and self.bounds["minY"] + radius < py < self.bounds["maxY"] - radius):
                return span - 5
            if self.ground(px, py) > z + 1.15 + span * .32:
                return span - 4
            for offset in (0, -.78, .78):
                sx, sy = px - uy * radius * offset, py + ux * radius * offset
                if self.solid(sx, sy, z + .85, z + height - .25):
                    return span - 5
            if span >= limit:
                return limit
        return limit

    def waypoint(self, position, goal, radius, height):
        """Find a nearby route around solid ground obstacles."""
        spacing = 7
        start = (round(position[0] / spacing), round(position[1] / spacing))
        target = (round(goal[0] / spacing), round(goal[1] / spacing))
        if start == target:
            return goal
        candidates = [(0, 0, start)]
        arrived = {start: (0, None)}
        best, best_dist = start, distance(start, target)
        expanded = 0
        offsets = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while candidates and expanded < 110:
            _, cost, current = heapq.heappop(candidates)
            if cost != arrived[current][0]:
                continue
            expanded += 1
            remaining = distance(current, target)
            if remaining < best_dist:
                best, best_dist = current, remaining
            if remaining <= 1 or distance(current, start) >= 11 and remaining < distance(start, target):
                break
            x, y = current[0] * spacing, current[1] * spacing
            z = position[2] if current == start else self.ground(x, y)
            for dx, dy in offsets:
                neighbor = current[0] + dx, current[1] + dy
                length = math.hypot(dx, dy)
                next_cost = cost + length
                estimated = next_cost + 1.12 * distance(neighbor, target)
                if next_cost >= arrived.get(neighbor, (math.inf, None))[0]:
                    continue
                span = spacing * length
                if self.clearance((x, y, z), dx / length, dy / length, radius, height, span) < span - 1:
                    continue
                arrived[neighbor] = (next_cost, current)
                heapq.heappush(candidates, (estimated, next_cost, neighbor))
        if best == start:
            return goal
        while arrived[best][1] != start:
            best = arrived[best][1]
        return [best[0] * spacing, best[1] * spacing]


class Pilot:
    name = "Oval Vector"

    def __init__(self, hello):
        self.config = hello["config"]
        self.map = Arena(hello["map"]["dir"])
        self.contact = -100.0
        self.last_bearing = None
        self.target_id = None
        self.turn_side = 1 if hello["slot"] % 2 == 0 else -1
        self.last_pos = None
        self.stuck = 0
        self.previous_move = [0, 0]
        self.scan_origin = hello["slot"] * .71
        self.waypoint = None
        self.route_goal = None
        self.route_tick = -100

    def zone_goal(self, obs, position):
        zone, now = obs["zone"], obs["t"]
        center, upcoming = zone["center"], zone["next_center"]
        radius = zone["radius"]
        future = zone["next_radius"]
        remaining = zone["shrink_end"] - now
        margin = max(1.4, min(12, radius * .2))
        future_distance = distance(position, upcoming)
        travel = max(0, future_distance - max(0, future - 8))
        urgency = (
            not obs["self"]["in_zone"] or distance(position, center) > radius - margin
            or travel / obs["self"]["stats"]["walk_speed"] + 9 > remaining
        )
        if radius < 35 or urgency:
            return upcoming, True
        # Stay within reach of the next circle while leaving room for local looting.
        if future_distance > future * .79:
            dx, dy = (position[i] - upcoming[i] for i in range(2))
            goal = [upcoming[0] + dx * future * .58 / future_distance,
                    upcoming[1] + dy * future * .58 / future_distance]
            return goal, False
        return upcoming, False

    def route(self, position, goal, stats, obs, opponent, retreat):
        dx, dy = goal[0] - position[0], goal[1] - position[1]
        dist = math.hypot(dx, dy)
        goal_yaw = math.atan2(dy, dx)
        if dist < 1:
            return [0., 0.], False
        radius, height = stats["radius"], stats["height"]
        zone = obs["zone"]
        zoner = zone["radius"]
        center = zone["center"]
        now_zone = distance(position, center)
        hostile = [
            projectile for projectile in obs["projectiles"]
            if distance(projectile["pos"], position) < 105 and abs(projectile["pos"][2] - position[2]) < 25
        ][:10]
        speed = stats["walk_speed"]
        best, direction, clearance = -1e9, (0, 0), 0
        for offset in (0, .28, -.28, .60, -.60, 1.03, -1.03, 1.48, -1.48, 2.05, -2.05, math.pi):
            a = goal_yaw + offset
            ux, uy = math.cos(a), math.sin(a)
            clear = self.map.clearance(position, ux, uy, radius, height, dist)
            score = 4.7 * math.cos(offset) + min(16, clear) * .27
            if clear < 4:
                score -= 10
            score += .52 * (ux * self.previous_move[0] + uy * self.previous_move[1])
            score += .26 * math.sin(offset) * self.turn_side
            endpoint = (position[0] + ux * 12, position[1] + uy * 12)
            zone_dist = distance(endpoint, center)
            score -= min(11, max(0, zone_dist - max(zoner - 2, now_zone)) * 1.7)
            if opponent is not None:
                o = opponent["pos"]
                edge = distance(endpoint, o)
                current = distance(position, o)
                if retreat:
                    score += clamp(edge - current, -8, 9) * .30
                    if clear > 5:
                        probe = (position[0] + ux * 10, position[1] + uy * 10, position[2] + height * .55)
                        if not self.map.ray((o[0], o[1], o[2] + self.config["mech"]["muzzle_height"]), probe):
                            score += 3.4
                yaw = opponent.get("aim", (0, 0))[0]
                bearing = math.atan2(endpoint[1] - o[1], endpoint[0] - o[0])
                alignment = abs(angle(bearing - yaw))
                if current < 105:
                    score += min(.45, alignment) * 2.3
            for projectile in hostile:
                p, v = projectile["pos"], projectile["vel"]
                rx, ry = p[0] - position[0], p[1] - position[1]
                cur_x, cur_y = obs["self"]["vel"][:2]
                vx, vy = v[0] - cur_x, v[1] - cur_y
                denom = vx * vx + vy * vy
                hit = -(rx * vx + ry * vy) / max(denom, .01)
                if not 0 < hit < .67:
                    continue
                gravity = self.config["weapons"]["mortar"]["gravity"] if projectile["kind"] == "mortar" else 0
                height_at_hit = p[2] + v[2] * hit - gravity * hit * hit * .5 - position[2]
                if -1 < height_at_hit < height + 1:
                    dvx, dvy = ux * speed - cur_x, uy * speed - cur_y
                    acceleration = stats["ground_accel"] if obs["self"]["grounded"] else stats["air_accel"]
                    ramp = min(hit, math.hypot(dvx, dvy) / acceleration)
                    advance = hit - .5 * ramp
                    if ramp >= hit:
                        advance *= min(1, acceleration * hit / max(.01, math.hypot(dvx, dvy)))
                    closest_x = rx + vx * hit - dvx * advance
                    closest_y = ry + vy * hit - dvy * advance
                    intercept_vx, intercept_vy = vx - dvx, vy - dvy
                    separation = abs(closest_x * intercept_vy - closest_y * intercept_vx) / max(
                        .01, math.hypot(intercept_vx, intercept_vy)
                    )
                    margin = radius + (6 if projectile["kind"] == "mortar" else 1.1)
                    score -= max(0, margin - separation) * 4.6
            if score > best:
                best, direction, clearance = score, (ux, uy), clear
        jump = False
        energy = obs["self"]["energy"]
        higher = len(goal) > 2 and goal[2] - position[2] > 2.5
        if energy > 23 and ((higher and dist < 26) or
                             (self.stuck > 7 and clearance < 8 and dist > 9)):
            jump = True
            if self.map.clearance(position, math.cos(goal_yaw), math.sin(goal_yaw), radius, height, dist) > 0:
                direction = (math.cos(goal_yaw), math.sin(goal_yaw))
        scale = min(1., dist / 5.0)
        return [direction[0] * scale, direction[1] * scale], jump

    def target(self, obs, me):
        position = me["pos"]
        muzzle = (*position[:2], position[2] + me["stats"]["muzzle_height"])
        rivals = []
        for enemy in obs["mechs"]:
            dist = distance(position, enemy["pos"])
            dz = enemy["pos"][2] - position[2]
            if dist > 250 or abs(dz) > 85:
                continue
            if enemy["kind"] == "npc":
                score = 35 - .3 * dist
            else:
                fraction = enemy.get("hp", 1400) / max(1, enemy.get("max_hp", 1400))
                score = 93 - .30 * dist + 23 * (1 - fraction)
                if enemy.get("charging"):
                    score += 9
                if enemy["id"] == self.target_id:
                    score += 8
            rivals.append((score, enemy))
        for crate in obs["crates"]:
            dist = distance(position, crate["pos"])
            if dist < 125:
                rivals.append((41 - .28 * dist + (300 - crate["hp"]) * .025, crate))
        rivals.sort(key=lambda pair: -pair[0])
        hidden = None
        for _, enemy in rivals[:8]:
            target = enemy["pos"]
            h = self.target_height(enemy)
            if self.map.ray(muzzle, (target[0], target[1], target[2] + h)):
                return enemy, True
            if enemy.get("kind") == "mech" and hidden is None:
                hidden = enemy
        return hidden, False

    def target_height(self, enemy):
        kind = enemy.get("kind")
        if kind == "mech":
            return self.config["mech"]["height"] * .56 * (1 + .05 * enemy.get("cubes", 0))
        if kind == "npc":
            return self.config["npc"]["height"] * .51
        return self.config["crates"]["height"] * .55

    def solution(self, weapon, me, enemy, dt):
        kind = weapon["kind"]
        stats, position = weapon["stats"], me["pos"]
        mortar = kind == "mortar"
        height = me["stats"]["height"] + .5 if mortar else me["stats"]["muzzle_height"]
        origin = position[0], position[1], position[2] + height
        target = enemy["pos"]
        velocity = enemy.get("vel", (0, 0, 0))
        target_vel = [velocity[i] for i in range(3)]
        if kind == "laser":
            flight = 0
        elif kind == "lightning":
            flight = 0
        else:
            flight = distance(origin, target) / stats["speed"]
        gravity = stats.get("gravity", 0) if mortar else 0
        for _ in range(6 if mortar else 3):
            lead = dt + flight
            point = [target[i] + target_vel[i] * lead for i in range(3)]
            floor = self.map.ground(point[0], point[1])
            point[2] = max(floor, point[2]) + self.target_height(enemy)
            dx, dy, dz = (b - a for a, b in zip(origin, point))
            planar = math.hypot(dx, dy)
            length = math.hypot(planar, dz)
            dz += gravity * flight * flight * .5
            pitch = math.atan2(dz, planar)
            if not flight:
                break
            next_flight = (math.hypot(planar, dz) - 2.8) / stats["speed"]
            flight = max(0, next_flight)
        if mortar and (length < 13 or flight > 2.7):
            return None
        if length > stats.get("range", 235) * (1 if kind != "lightning" else .96):
            return None
        if not self.map.ray(origin, point, gravity, flight):
            return None
        return math.atan2(dy, dx), pitch, length

    def fight(self, me, enemy, action, dt):
        weapons = me["weapons"]
        laser_charging = any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in weapons)
        if enemy is None:
            action["reload"] = any(w and w["kind"] == "gun" and
                                   w.get("ammo", 99) < w["stats"]["mag"] * .8 for w in weapons)
            return
        options = []
        for index, weapon in enumerate(weapons):
            if not weapon:
                continue
            kind = weapon["kind"]
            solution = self.solution(weapon, me, enemy, dt * .60)
            if solution is None:
                continue
            yaw, pitch, length = solution
            stats = weapon["stats"]
            if not me["stats"]["min_pitch"] <= pitch <= me["stats"]["max_pitch"]:
                continue
            weight = {"gun": 25, "mortar": 54, "laser": 74, "lightning": 55}[kind]
            cooldown = weapon.get("cooldown", 0)
            if kind == "gun" and (weapon.get("reload", 0) > .08 or weapon.get("ammo", 1) == 0):
                weight -= 35
            if kind == "mortar":
                weight -= max(0, length - 58) * .34
            if kind == "lightning" and me["energy"] <= stats["energy"] + 4:
                weight -= 60
            if kind == "laser" and (charge := weapon.get("charge", 0)) > 0:
                # Use the arm gun while building charge, then prioritize the beam.
                weight = 18 if charge + dt * 1.7 < stats["charge"] else 140
            elif cooldown > .05:
                weight -= min(150, cooldown * 180)
            options.append((weight, index, weapon, yaw, pitch, length))
        if not options:
            if laser_charging:
                for i, w in enumerate(weapons):
                    if w and w["kind"] == "laser":
                        action["fire"][i] = True
            aim = self.direct_aim(me, enemy)
            action["aim"] = aim
            return
        _, selected, primary, yaw, pitch, length = max(options, key=lambda option: option[0])
        action["aim"] = [yaw, pitch]
        actual_yaw, actual_pitch = me["aim"]
        shift_yaw = me["stats"]["turn_rate"] * dt * .45
        shift_pitch = me["stats"]["pitch_rate"] * dt * .45
        yaw_error = max(0, abs(angle(yaw - actual_yaw)) - shift_yaw)
        pitch_error = max(0, abs(pitch - actual_pitch) - shift_pitch)
        tolerance = clamp(2.0 / max(length, 1), .012, .29)
        if enemy.get("kind") is None:
            tolerance *= .48
        for _, i, w, y, p, d in options:
            kind = w["kind"]
            aligned = yaw_error < tolerance and pitch_error < tolerance * 1.22
            same = abs(angle(y - yaw)) < tolerance * .62 and abs(p - pitch) < tolerance * .74
            if kind == "lightning":
                lined_up = abs(angle(y - actual_yaw)) < math.radians(w["stats"]["cone_deg"]) * .42
                action["fire"][i] = lined_up and me["energy"] > w["stats"]["energy"] + 5
            elif kind == "laser":
                charge = w.get("charge", 0)
                desired = w["stats"]["charge"] - dt * .35
                action["fire"][i] = (i == selected and (charge < desired or not aligned)) or (
                    charge > 0 and (not same or not aligned or charge < desired)
                )
            elif i == selected or same:
                action["fire"][i] = aligned and w.get("cooldown", 0) < dt + .03

    def direct_aim(self, me, enemy):
        position, target = me["pos"], enemy["pos"]
        dx, dy = target[0] - position[0], target[1] - position[1]
        dz = target[2] + self.target_height(enemy) - position[2] - me["stats"]["muzzle_height"]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def loot(self, obs, me, action, zone_goal, danger):
        position, zone = me["pos"], obs["zone"]
        options = []
        held = [w["kind"] if w else None for w in me["weapons"]]
        drop_slot = None
        if None not in held:
            secondary = 1 if held[0] == "gun" else 0
            drop_slot = secondary
        old_value = PICKUP_VALUES.get(held[drop_slot], 0) if drop_slot is not None else 0
        for pickup in obs["pickups"]:
            kind = pickup["kind"]
            value = PICKUP_VALUES.get(kind, 0)
            if kind in held or value <= old_value:
                continue
            p = pickup["pos"]
            dist = distance(position, p)
            if dist < self.config["loot"]["pickup_radius"] * .98 and abs(p[2] - position[2]) < 3.9:
                action["take"] = pickup["id"]
                if drop_slot is not None:
                    action["drop"] = drop_slot
                break
            margin = zone["radius"] - distance(p, zone["center"])
            score = 35 + 9 * value - dist * .31 - abs(p[2] - position[2]) * .42
            if dist < 120 and margin > 6:
                options.append((score, p))
        for cube in obs["cubes"]:
            p = cube["pos"]
            d = distance(position, p)
            if d < 95 and distance(p, zone["center"]) < zone["radius"] - 3:
                options.append((68 - d * .47 - abs(p[2] - position[2]) * .72, p))
        for drop in obs.get("drops", ()):
            p = drop["pos"]
            d = distance(position, p)
            if drop["lands_in"] > min(18., d / 9 + 4) or d > 190:
                continue
            if distance(p, zone["center"]) < zone["radius"] - 8:
                injury = 1 - me["hp"] / me["max_hp"]
                options.append((90 + 45 * injury - .35 * d, p))
        if not danger:
            for crate in obs["crates"]:
                p = crate["pos"]
                d = distance(position, p)
                if 30 < d < 135 and distance(p, zone["center"]) < zone["radius"] - 12:
                    options.append((33 - d * .16 - abs(p[2] - position[2]) * .35, p))
        return max(options, key=lambda entry: entry[0]) if options else (0, zone_goal)

    def act(self, obs):
        me, now = obs["self"], obs["t"]
        position, stats = me["pos"], me["stats"]
        dt = self.config["match"]["decide_every"] / self.config["match"]["tick_hz"]
        action = {"move": [0, 0], "aim": me["aim"], "fire": [False, False]}
        if self.last_pos is not None and math.hypot(*self.previous_move) > .5:
            self.stuck = self.stuck + 1 if distance(position, self.last_pos) < .25 else max(0, self.stuck - 2)
        self.last_pos = position
        for event in obs["events"]:
            if event["type"] in ("hit", "damaged") and event.get("weapon") != "zone":
                self.contact = now
                if event["type"] == "damaged" and event.get("bearing") is not None:
                    self.last_bearing = event["bearing"]
        if offer := obs.get("offer"):
            action["pick"] = max(range(len(offer["choices"])),
                                 key=lambda i: UPGRADE_VALUES.get(offer["choices"][i]["id"], 0))

        target, visible = self.target(obs, me)
        engage = target is not None and target.get("kind") == "mech"
        enemy_distance = distance(position, target["pos"]) if engage else math.inf
        hostile = [e for e in obs["mechs"] if e["kind"] == "mech" and distance(position, e["pos"]) < 115]
        hp_fraction = me["hp"] / me["max_hp"]
        enemy_fraction = target.get("hp", 1400) / max(1, target.get("max_hp", 1400)) if engage else 1
        retreat = engage and (hp_fraction < .28 or (hp_fraction < .51 and enemy_fraction > hp_fraction * 1.3)
                              or (hp_fraction < .66 and len(hostile) >= 2))
        held = {w["kind"] for w in me["weapons"] if w}
        future, urgent = self.zone_goal(obs, position)
        value, goal = self.loot(obs, me, action, future, bool(hostile))
        if urgent:
            goal = future
        elif engage:
            enemy_pos = target["pos"]
            dist = max(1, enemy_distance)
            ux, uy = (enemy_pos[0] - position[0]) / dist, (enemy_pos[1] - position[1]) / dist
            holds_laser, holds_melee = "laser" in held, "lightning" in held
            desired = 78 if holds_laser else 36 if holds_melee else 52
            if any(w == "lightning" for w in target.get("weapons", [])) and not holds_melee:
                desired = max(desired, 68)
            radial = clamp((dist - desired) / 28, -1, 1)
            if retreat:
                radial = -1.7
            else:
                radial += .27 * math.sin(now * 3.6 + self.scan_origin)
            side = self.turn_side
            if dist > 95 and value > 49 and not holds_laser and not retreat:
                pass  # Arm the mech while the opponent is still at long range.
            elif not visible and not retreat and dist > 30:
                goal = enemy_pos if value < 55 else goal
            else:
                goal = [position[0] + 24 * (radial * ux + side * uy),
                        position[1] + 24 * (radial * uy - side * ux)]
        elif target is not None and visible and value < 38:
            # Hold a good loot-farming range and close the pickup distance quickly.
            target_distance = distance(position, target["pos"])
            desired = 32 if target.get("kind") == "npc" else 18
            if target_distance > desired:
                fraction = 1 - desired / max(desired, target_distance)
                goal = [position[i] + fraction * (target["pos"][i] - position[i]) for i in range(2)]
            else:
                goal = position[:2]
        elif value < 1 and distance(position, future) < min(50, obs["zone"]["radius"] * .34):
            heading = self.scan_origin + now * .085
            span = min(45, obs["zone"]["next_radius"] * .5)
            goal = [future[0] + math.cos(heading) * span, future[1] + math.sin(heading) * span]

        movement_goal = goal
        if (not engage or urgent) and distance(position, goal) > 12 and (
            len(goal) == 2 or goal[2] <= position[2] + 2.5
        ) and position[2] < self.map.ground(*position[:2]) + 2:
            if obs["tick"] - self.route_tick >= 12:
                self.route_tick = obs["tick"]
                dx, dy = goal[0] - position[0], goal[1] - position[1]
                length = math.hypot(dx, dy)
                cut = self.map.clearance(position, dx / length, dy / length, stats["radius"], stats["height"], length)
                self.waypoint = (
                    self.map.waypoint(position, goal, stats["radius"], stats["height"])
                    if cut < min(length - 1, 16) else None
                )
                self.route_goal = goal
            if self.waypoint and distance(goal, self.route_goal) < 15 and distance(position, self.waypoint) > 2:
                movement_goal = self.waypoint
        move, jump = self.route(position, movement_goal, stats, obs, target if engage else None, retreat)
        self.previous_move = move
        action["move"] = move
        action["jump"] = bool(jump and me["grounded"])
        energy = me["energy"]
        if me["grounded"]:
            action["boost"] = energy > (14 if urgent or retreat else 57) and (
                urgent or retreat or (distance(position, goal) > 45 and not engage)
            ) and math.hypot(*move) > .7
        else:
            roof_ascent = len(goal) > 2 and goal[2] - position[2] > .8
            action["boost"] = energy > 8 and (roof_ascent or jump)

        if retreat and (not visible or enemy_distance > 52 or hp_fraction < .20):
            self.fight(me, None, action, dt)
            action["aim"] = self.direct_aim(me, target)
        elif target is not None:
            self.fight(me, target, action, dt)
        else:
            self.fight(me, None, action, dt)
            heading = math.atan2(move[1], move[0]) if math.hypot(*move) > .2 else self.scan_origin
            if self.last_bearing is not None and now - self.contact < 1.5:
                heading = self.last_bearing
            else:
                heading += math.sin(now * 1.47 + self.scan_origin) * .96
            action["aim"] = [angle(heading), .01]
        self.target_id = target["id"] if target is not None else None
        return action


if __name__ == "__main__":
    run(Pilot)
