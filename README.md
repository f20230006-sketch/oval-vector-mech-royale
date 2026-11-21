# Oval Vector

Python bot for the Builders Mech Royale track. The bot reads the official arena at `hello.map.dir` and responds over the `mechfight/1` protocol.

The pilot leads moving targets, alternates gun fire with laser charging, estimates which movement dodges incoming projectiles, navigates around buildings, and weighs weapon pickups and supply drops against zone pressure.

The pilot remembers recent attackers and weapon pickups, prioritizes finishing exposed opponents, and uses terrain to break contact when outmatched. It can use its jets to reach a roof with a clear firing lane, and abandons the climb when the zone or threats make it unsafe.

```sh
docker build --platform linux/amd64 -t oval-vector .
```

Run the official kit's conformance checker with the `oval-vector` image and the kit arena mounted as `/arena`. Submit this repository once the checker passes. Practice placement comes from the contest's 48 private matches. [Mech Royale track](https://www.joinbuilders.org/buildathon/tracks/builders)
