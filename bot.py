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
PICKUP_VALUES = {"gun": 0, "mortar": 2, "laser": 4, "lightning": 5}
UPGRADE_VALUES = {
    "nanites": 15, "plating": 14, "armor": 13, "servos": 12, "reactor": 11,
    "laser_charge": 10.5, "laser_damage": 10, "jets": 9.5, "gun_mag": 8.5,
    "gun_damage": 8, "gun_rate": 7.5, "gun_velocity": 7, "lightning_range": 7,
    "lightning_efficiency": 6.5, "mortar_velocity": 5, "mortar_splash": 4.5,
    "mortar_reload": 4, "gyros": 4, "capacitor": 4, "sensors": 3,
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
        self.ledges = []
        for solid in world["prisms"]:
            xs, ys = solid["ring"][::2], solid["ring"][1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            item = (x0, x1, y0, y1, solid["base"], solid["top"],
                    polygon(solid["ring"]), [polygon(hole) for hole in solid["holes"]])
            for ix in range(math.floor(x0 / 16), math.floor(x1 / 16) + 1):
                for iy in range(math.floor(y0 / 16), math.floor(y1 / 16) + 1):
                    self.bins[ix, iy].append(item)
            if solid["kind"] == "building" and 8 <= solid["top"] - solid["base"] <= 30:
                self.add_ledges(item)

    def add_ledges(self, building):
        _, _, _, _, _, top, ring, holes = building
        for x1, y1, x2, y2 in ring:
            span = math.hypot(x2 - x1, y2 - y1)
            if span < 13:
                continue
            mx, my = (x1 + x2) * .5, (y1 + y2) * .5
            nx, ny = (y1 - y2) / span, (x2 - x1) / span
            if not inside(mx + nx, my + ny, ring):
                nx, ny = -nx, -ny
            landing = (mx + nx * 5.5, my + ny * 5.5, top)
            if not inside(*landing[:2], ring) or any(inside(*landing[:2], hole) for hole in holes):
                continue
            takeoff = (mx - nx * 7, my - ny * 7)
            self.ledges.append((takeoff, landing))

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
        self.last_attacker = None
        self.last_damage = -100.0
        self.recovering = False
        self.pressure = 0.0
        self.contacts = {}
        self.loot_memory = {}
        self.climb = None
        self.climb_started = -100.0
        self.climb_blocked_until = -100.0

    def find_perch(self, obs, me, enemy):
        position, zone, stats = me["pos"], obs["zone"], me["stats"]
        limit = min(31, max(0, me["energy"] - 25) / stats["jet_drain"] * stats["max_rise_speed"] - 4)
        target = enemy["pos"]
        best, ledge = -math.inf, None
        for takeoff, landing in self.map.ledges:
            approach = distance(position, takeoff)
            rise = landing[2] - position[2]
            separation = distance(landing, target)
            if (
                approach > 25 or not 7 < rise < limit or not 36 < separation < 135
                or distance(landing, zone["center"]) > zone["radius"] - 9
                or distance(landing, zone["next_center"]) > zone["next_radius"] - 8
            ):
                continue
            low = self.map.ground(*takeoff)
            if not self.map.ray((*position[:2], position[2] + 4), (*takeoff, low + 4)):
                continue
            if self.map.solid(*takeoff, low + .5, landing[2] + stats["height"]):
                continue
            if self.map.solid(*landing[:2], landing[2] + .3, landing[2] + stats["height"]):
                continue
            muzzle = (*landing[:2], landing[2] + stats["muzzle_height"])
            aim = (*target[:2], target[2] + self.target_height(enemy))
            if not self.map.ray(muzzle, aim):
                continue
            score = 1.8 * min(18, landing[2] - target[2]) - approach - rise * .34
            if score > best:
                best, ledge = score, (takeoff, landing)
        return ledge

    def perch_route(self, obs, me, enemy, urgent, retreat, held):
        """Commit to a short open-air ascent before crossing the roof edge."""
        now = obs["t"]
        position = me["pos"]
        if urgent or retreat or enemy is None:
            self.climb = None
            return None
        if self.climb is None and now >= self.climb_blocked_until:
            dist = distance(position, enemy["pos"])
            if (
                not me["grounded"] or not 47 < dist < 130 or me["energy"] < 72
                or "laser" not in held or len(self.contacts) > 1
                or self.pressure > 95 or self.map.ground(*position[:2]) + 3 < position[2]
            ):
                return None
            self.climb = self.find_perch(obs, me, enemy)
            self.climb_started = now
            self.climb_blocked_until = now + 2
        if self.climb is None:
            return None
        takeoff, landing = self.climb
        if now - self.climb_started > 6 or me["energy"] < 8:
            self.climb = None
            self.climb_blocked_until = now + 6
            return None
        if position[2] >= landing[2] + .15:
            if distance(position, landing) > 2.5:
                return landing, False, not me["grounded"] and position[2] < landing[2] + 5
            self.climb = None
            return None
        if me["grounded"] and distance(position, takeoff) > 3:
            return takeoff, False, False
        return takeoff, me["grounded"], not me["grounded"]

    def observe(self, obs, me):
        now = obs["t"]
        self.pressure *= .94
        visible = {}
        for enemy in obs["mechs"]:
            if enemy["kind"] == "mech":
                self.contacts[enemy["id"]] = (now, enemy)
                visible[enemy["id"]] = enemy
        for event in obs["events"]:
            kind = event["type"]
            if kind == "kill":
                self.contacts.pop(event["target"], None)
            elif kind == "damaged" and event.get("weapon") != "zone":
                self.contact = self.last_damage = now
                self.pressure += event["amount"]
                if event.get("by") is not None:
                    self.last_attacker = event["by"]
                if event.get("bearing") is not None:
                    self.last_bearing = event["bearing"]
            elif kind == "hit":
                self.contact = now

        position = me["pos"]
        threats = []
        for seen, enemy in self.contacts.values():
            age = now - seen
            if age > 3.5:
                continue
            pos = enemy["pos"]
            if enemy["id"] not in visible:
                dx, dy, _ = enemy.get("vel", (0, 0, 0))
                pos = [pos[0] + dx * min(age, .8), pos[1] + dy * min(age, .8), pos[2]]
            dist = distance(position, pos)
            if dist > 185:
                continue
            bearing = math.atan2(position[1] - pos[1], position[0] - pos[0])
            facing = abs(angle(bearing - enemy.get("aim", (bearing, 0))[0]))
            weight = max(0, 1.35 - dist / 150) * (1 if facing < .78 else .47)
            if enemy.get("charging"):
                weight += .30
            if "lightning" in enemy.get("weapons", ()) and dist < 54:
                weight += .40
            if enemy["id"] == self.last_attacker and now - self.last_damage < 2.0:
                weight += .60
            weight *= max(.25, 1 - age * .32)
            if weight > .18:
                threats.append((weight, pos))

        self.contacts = {key: pair for key, pair in self.contacts.items() if now - pair[0] <= 3.5}
        threats.sort(reverse=True, key=lambda item: item[0])
        return threats[:4]

    def escape_goal(self, obs, me, threats, zone_goal):
        position, stats, zone = me["pos"], me["stats"], obs["zone"]
        if threats:
            origin = threats[0][1]
            heading = math.atan2(position[1] - origin[1], position[0] - origin[0])
        elif self.last_bearing is not None:
            heading = self.last_bearing + math.pi
        else:
            return zone_goal

        origin_zone = distance(position, zone["center"])
        result = zone_goal
        best = -math.inf
        for offset in (0, .35, -.35, .7, -.7, 1.15, -1.15, 1.65, -1.65, 2.2, -2.2):
            direction = heading + offset
            dx, dy = math.cos(direction), math.sin(direction)
            clearance = self.map.clearance(position, dx, dy, stats["radius"], stats["height"], 22)
            if clearance < 5:
                continue
            travel = min(23, clearance)
            x, y = position[0] + dx * travel, position[1] + dy * travel
            end = (x, y, self.map.ground(x, y) + stats["height"] * .54)
            score = 4 * math.cos(offset) + .16 * travel
            score += .75 * (dx * self.previous_move[0] + dy * self.previous_move[1])
            score -= max(0, distance(end, zone["center"]) - max(origin_zone, zone["radius"] - 5)) * 2
            for threat, source in threats[:3]:
                muzzle = (*source[:2], source[2] + self.config["mech"]["muzzle_height"])
                separation = distance(end, source) - distance(position, source)
                score += threat * (clamp(separation * .21, -5, 4) +
                                   (5 if not self.map.ray(muzzle, end) else 0))
            if score > best:
                best, result = score, end[:2]
        return result

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
        effective_range = max((
            weapon["stats"].get("range", 150) for weapon in me["weapons"] if weapon
        ), default=0)
        arming = any(weapon is None for weapon in me["weapons"]) and self.pressure < 105
        enemy_nearby = any(
            enemy["kind"] == "mech" and distance(position, enemy["pos"]) < 60
            for enemy in obs["mechs"]
        )
        rivals = []
        for enemy in obs["mechs"]:
            dist = distance(position, enemy["pos"])
            dz = enemy["pos"][2] - position[2]
            if dist > 250 or abs(dz) > 85:
                continue
            if enemy["kind"] == "npc":
                score = 47 - .33 * dist
                if me["cubes"] < 6:
                    score += 8
                score += 14 * (1 - enemy["hp"] / max(1, enemy["max_hp"]))
            else:
                fraction = enemy.get("hp", 1400) / max(1, enemy.get("max_hp", 1400))
                bearing = math.atan2(enemy["pos"][1] - position[1], enemy["pos"][0] - position[0])
                turn = abs(angle(bearing - me["aim"][0]))
                score = 87 - .36 * dist + 44 * (1 - fraction) - 7 * turn
                if dist > effective_range or (dist > 115 and fraction > .55):
                    score -= 28
                if enemy.get("charging"):
                    score += 5
                if enemy["id"] == self.last_attacker and obs["t"] - self.last_damage < 1.6:
                    score += 17
                if .0 < fraction < .20:
                    score += 14
                if enemy["id"] == self.target_id:
                    score += 8
            rivals.append((score, enemy))
        for crate in obs["crates"]:
            dist = distance(position, crate["pos"])
            if dist < 125:
                score = 41 - .28 * dist + (300 - crate["hp"]) * .025
                if arming and not enemy_nearby and dist < 42:
                    score += 26
                rivals.append((score, crate))
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
            cubes = self.config["cubes"]
            growth = cubes["size_per_cube"] * enemy.get("cubes", 0)
            return self.config["mech"]["height"] * .56 * (1 + growth)
        if kind == "npc":
            return self.config["npc"]["height"] * .51
        return self.config["crates"]["height"] * .55

    def solution(self, weapon, me, enemy, dt):
        kind = weapon["kind"]
        stats, position = weapon["stats"], me["pos"]
        mortar = kind == "mortar"
        height = me["stats"]["height"] + .5 if mortar else me["stats"]["muzzle_height"]
        shooter_velocity = me.get("vel", (0, 0, 0))
        origin = (
            position[0] + shooter_velocity[0] * dt,
            position[1] + shooter_velocity[1] * dt,
            position[2] + height + shooter_velocity[2] * dt,
        )
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
            weight = {"gun": 25, "mortar": 54, "laser": 74, "lightning": 67}[kind]
            cooldown = weapon.get("cooldown", 0)
            if kind == "gun" and (weapon.get("reload", 0) > .08 or weapon.get("ammo", 1) == 0):
                weight -= 35
            if kind == "mortar":
                weight -= max(0, length - 58) * .27
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
        _, selected, primary, yaw, pitch, _ = max(options, key=lambda option: option[0])
        if primary["kind"] == "lightning":
            cone = math.radians(primary["stats"]["cone_deg"]) * .38
            for _, _, weapon, gun_yaw, gun_pitch, _ in options:
                if (
                    weapon["kind"] == "gun" and weapon.get("ammo", 0) > 0
                    and weapon.get("reload", 0) <= dt * .5
                    and weapon.get("cooldown", 0) <= dt
                    and abs(angle(gun_yaw - yaw)) < cone
                ):
                    # The beam has a yaw cone. Aim for the smaller gun window inside it.
                    yaw, pitch = gun_yaw, gun_pitch
                    break
        action["aim"] = [yaw, pitch]
        actual_yaw, actual_pitch = me["aim"]
        for _, i, w, y, p, d in options:
            kind = w["kind"]
            interval = dt * (.95 if w.get("cooldown", 0) > dt * .5 else .49)
            turn = me["stats"]["turn_rate"] * interval
            tilt = me["stats"]["pitch_rate"] * interval
            firing_yaw = actual_yaw + clamp(angle(yaw - actual_yaw), -turn, turn)
            firing_pitch = actual_pitch + clamp(pitch - actual_pitch, -tilt, tilt)
            tolerance = clamp(2.0 / max(d, 1), .009, .29)
            if enemy.get("kind") is None:
                tolerance *= .58
            yaw_error = abs(angle(y - firing_yaw))
            pitch_error = abs(p - firing_pitch)
            aligned = yaw_error < tolerance and pitch_error < tolerance * 1.32
            same = abs(angle(y - yaw)) < tolerance and abs(p - pitch) < tolerance * 1.2
            if kind == "lightning":
                lined_up = yaw_error < math.radians(w["stats"]["cone_deg"]) * .43
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
        now = obs["t"]
        self.loot_memory.update((item["id"], (now, item)) for item in obs["pickups"])
        for key, (seen, item) in list(self.loot_memory.items()):
            if now - seen > 12 or (
                now != seen and distance(position, item["pos"]) < 6
                and abs(position[2] - item["pos"][2]) < 4
            ):
                del self.loot_memory[key]
        held = [w["kind"] if w else None for w in me["weapons"]]
        drop_slot = None
        if None not in held:
            secondary = 1 if held[0] == "gun" else 0
            drop_slot = secondary
        old_value = PICKUP_VALUES.get(held[drop_slot], 0) if drop_slot is not None else 0
        nearby_weapon = None
        for seen, pickup in self.loot_memory.values():
            kind = pickup["kind"]
            value = PICKUP_VALUES.get(kind, 0)
            if kind in held or value <= old_value:
                continue
            p = pickup["pos"]
            dist = distance(position, p)
            if now == seen and dist < self.config["loot"]["pickup_radius"] * .98 and abs(p[2] - position[2]) < 3.9:
                if nearby_weapon is None or value > nearby_weapon[0]:
                    nearby_weapon = value, pickup["id"]
            margin = zone["radius"] - distance(p, zone["center"])
            score = 41 + 9 * value - dist * .33 - abs(p[2] - position[2]) * .60
            if None in held:
                score += 16
            if dist < 120 and margin > 6:
                options.append((score - (now - seen) * 1.6, p))
        if nearby_weapon is not None:
            action["take"] = nearby_weapon[1]
            if drop_slot is not None:
                action["drop"] = drop_slot
        cube_bins = defaultdict(int)
        for cube in obs["cubes"]:
            x, y, _ = cube["pos"]
            cube_bins[round(x / 7.5), round(y / 7.5)] += 1
        for cube in obs["cubes"]:
            p = cube["pos"]
            d = distance(position, p)
            if d < 95 and distance(p, zone["center"]) < zone["radius"] - 3:
                count = cube_bins[round(p[0] / 7.5), round(p[1] / 7.5)]
                growth = 10 if me["cubes"] < self.config["cubes"]["stat_cap"] else 0
                options.append((68 + growth + 9 * min(6, count - 1) - d * .47
                                - abs(p[2] - position[2]) * .72, p))
        for drop in obs.get("drops", ()):
            p = drop["pos"]
            d = distance(position, p)
            if drop["lands_in"] > min(18., d / 9 + 4) or d > 190:
                continue
            if distance(p, zone["center"]) < zone["radius"] - 8:
                injury = 1 - me["hp"] / me["max_hp"]
                committed = drop["holder"] == me["id"]
                options.append((91 + 78 * injury + 38 * committed - .43 * d, p))
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
        threats = self.observe(obs, me)
        if offer := obs.get("offer"):
            action["pick"] = max(range(len(offer["choices"])),
                                 key=lambda i: UPGRADE_VALUES.get(offer["choices"][i]["id"], 0))

        target, visible = self.target(obs, me)
        engage = target is not None and target.get("kind") == "mech"
        enemy_distance = distance(position, target["pos"]) if engage else math.inf
        alive_count = obs.get("alive", 16)
        conservative = alive_count > 3
        hp_fraction = me["hp"] / me["max_hp"]
        enemy_fraction = target.get("hp", 1400) / max(1, target.get("max_hp", 1400)) if engage else 1
        held = {w["kind"] for w in me["weapons"] if w}
        close_finish = engage and visible and enemy_distance < 90 and (
            target.get("hp", math.inf) <= (95 if "laser" in held else 52)
        )
        outmatched = (
            hp_fraction < .50
            or hp_fraction < .68 and len(threats) > 1
            or (conservative and (hp_fraction < .75 and enemy_fraction > hp_fraction - .12))
            or hp_fraction < .60 and self.pressure > 110
        )
        if hp_fraction >= .80 or (now - self.last_damage > 11 and hp_fraction >= .66):
            self.recovering = False
        elif outmatched and (threats or now - self.last_damage < 2.5):
            self.recovering = True
        retreat = self.recovering and not close_finish
        future, urgent = self.zone_goal(obs, position)
        value, goal = self.loot(obs, me, action, future, bool(threats))
        capture = next((
            drop for drop in obs.get("drops", ())
            if drop["holder"] == me["id"] and drop["lands_in"] <= 0
            and drop["capture_s"] - drop["progress"] <= 1.35
        ), None)
        if urgent:
            goal = future
        elif capture is not None:
            goal = capture["pos"]
        elif retreat and (threats or now - self.last_damage < 2.5):
            goal = self.escape_goal(obs, me, threats, future)
        elif retreat:
            goal = goal if value > 100 else position
        elif engage:
            enemy_pos = target["pos"]
            dist = max(1, enemy_distance)
            ux, uy = (enemy_pos[0] - position[0]) / dist, (enemy_pos[1] - position[1]) / dist
            holds_laser, holds_melee = "laser" in held, "lightning" in held
            desired = 90 if holds_laser else (45 if conservative else 30) if holds_melee else (65 if conservative else 51)
            if holds_melee and me["energy"] < 16:
                desired = 50
            if any(w == "lightning" for w in target.get("weapons", [])) and not holds_melee:
                desired = max(desired, 75)
            radial = clamp((dist - desired) / 28, -1, 1)
            radial += .23 * math.sin(now * 3.6 + self.scan_origin)
            side = self.turn_side
            reward_nearby = value > 88 and distance(position, goal) < 22 and hp_fraction > .40
            if not close_finish and (
                reward_nearby or dist > 60 and value > (50 if len(held) == 1 else 85)
            ):
                pass  # Equip or repair while the opponent is still at range.
            elif not visible:
                goal = goal if value > 25 else future
            else:
                goal = [position[0] + 24 * (radial * ux + side * uy),
                        position[1] + 24 * (radial * uy - side * ux)]
        elif target is not None and visible and value < 38:
            # Hold a good loot-farming range and close the pickup distance quickly.
            target_distance = distance(position, target["pos"])
            desired = 50 if target.get("kind") == "npc" else 18
            if target_distance > desired:
                fraction = 1 - desired / max(desired, target_distance)
                goal = [position[i] + fraction * (target["pos"][i] - position[i]) for i in range(2)]
            else:
                goal = position[:2]
        elif value < 1 and distance(position, future) < min(50, obs["zone"]["radius"] * .34):
            heading = self.scan_origin + now * .085
            span = min(45, obs["zone"]["next_radius"] * .5)
            goal = [future[0] + math.cos(heading) * span, future[1] + math.sin(heading) * span]

        climb = self.perch_route(
            obs, me, target if engage and capture is None and value < 88 else None, urgent, retreat, held
        )
        if climb is not None:
            goal, _, _ = climb
        elif engage and not urgent and me["grounded"]:
            # Keep a roof position while it offers a shot. A descent costs both cover and energy.
            floor = self.map.ground(*position[:2])
            if position[2] > floor + 5 and visible and not retreat and value < 106:
                goal = position

        movement_goal = goal
        if (not engage or urgent or retreat) and distance(position, goal) > 12 and (
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
        if climb is not None:
            _, action["jump"], action["boost"] = climb
            if isinstance(goal, tuple) and len(goal) == 3 and position[2] >= goal[2] + .15:
                dx, dy = goal[0] - position[0], goal[1] - position[1]
                span = math.hypot(dx, dy)
                if span > .4:
                    action["move"] = self.previous_move = [dx / span, dy / span]

        if retreat and engage and visible and enemy_distance < 78 and now - self.last_damage < .7:
            self.fight(me, target, action, dt)
        elif retreat:
            self.fight(me, None, action, dt)
            if target is not None and engage:
                action["aim"] = self.direct_aim(me, target)
            elif self.last_bearing is not None and now - self.last_damage < 3:
                action["aim"] = [self.last_bearing, 0.]
        elif target is not None:
            self.fight(me, target, action, dt)
        else:
            self.fight(me, None, action, dt)
            heading = math.atan2(move[1], move[0]) if math.hypot(*move) > .2 else self.scan_origin
            if self.last_bearing is not None and now - self.contact < 1.5:
                heading = self.last_bearing
            elif sounds := [sound for sound in obs.get("sounds", ())
                            if sound["kind"] in ("gun", "mortar", "lightning", "laser")
                            and sound["dist"] == "near" and sound["age"] < .5]:
                heading = min(sounds, key=lambda sound: sound["age"])["bearing"]
            else:
                heading += math.sin(now * 1.47 + self.scan_origin) * .96
            action["aim"] = [angle(heading), .01]
        if climb is not None:
            for i, weapon in enumerate(me["weapons"]):
                if weapon and weapon["kind"] == "laser":
                    action["fire"][i] = False
        self.target_id = target["id"] if target is not None else None
        return action


if __name__ == "__main__":
    run(Pilot)
