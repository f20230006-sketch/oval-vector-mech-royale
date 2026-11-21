# Oval Vector

Python bot for the Builders Mech Royale track. The bot reads the official arena at `hello.map.dir` and responds over the `mechfight/1` protocol.

The pilot leads moving targets, alternates gun fire with laser charging, estimates which movement dodges incoming projectiles, navigates around buildings, and weighs weapon pickups and supply drops against zone pressure.

The second practice build remembers recent attackers and weapon pickups. It prioritizes finishing exposed opponents, uses terrain to break contact when outmatched, then pauses fire to let health regeneration start. It passed all 300 observations in the official Docker conformance check.

```sh
docker build --platform linux/amd64 -t oval-vector .
```

Run the official kit's conformance checker with the `oval-vector` image and the kit arena mounted as `/arena`. Submit this repository once the checker passes. Practice placement comes from the contest's 48 private matches. [Mech Royale track](https://www.joinbuilders.org/buildathon/tracks/builders)
