# Ubuntu IRC Human Behavior Benchmark

This report compares the current wait-policy model against real timestamped human dialogue from the official Ubuntu IRC logs.

## Label Definition

- Data source: official Ubuntu IRC HTML logs
- Channel: `#ubuntu`
- Dates: `2026-02-04, 2026-02-05, 2026-02-06`
- Timestamp resolution: minute-level, as provided by the public logs
- Focal speaker: the nick who produced the anchor turn
- Positive label: delay until that same speaker speaks again within the observation window
- Suppress label: no same-speaker next turn within the observation window
- Role mapping for the current model: focal speaker -> `assistant`, everyone else -> `user`

This is a behavior-comparison benchmark. It measures how well the model matches actual human next-turn timing, not an explicit optimal-policy label.

## Aggregate Metrics

- Total cases: `164`
- Follow-up cases: `113`
- Suppress cases: `51`
- Overall within-tolerance rate: `0.1707`
- Follow-up within-tolerance rate: `0.2212`
- Suppress agreement rate: `0.0588`
- Follow-up detection rate: `0.9823`
- Follow-up MAE: `1673.9s`
- Follow-up median absolute error: `274.0s`
- Follow-up Spearman rank correlation: `0.0661`

## Visualization

![Ubuntu IRC human behavior comparison](ubuntu_irc_behavior_comparison.svg)

## Largest Misses

| Anchor time | Nick | Human | Model | Error | Source |
| --- | --- | --- | --- | --- | --- |
| `2026-02-06T16:26` | `SuperLag -> lotuspsychje` | `5.9h` | `37s` | `5.9h` | `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html` |
| `2026-02-04T18:41` | `JanC -> osse` | `5.2h` | `25s` | `5.1h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-06T13:05` | `tomreyn -> ajorj` | `4.8h` | `26s` | `4.8h` | `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html` |
| `2026-02-06T13:07` | `tomreyn -> TomyWork` | `4.8h` | `40s` | `4.8h` | `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html` |
| `2026-02-04T14:17` | `TomyWork -> leftyfb` | `2.0h` | `36s` | `2.0h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:23` | `hansolefsen2012 -> lazysundaydreams` | `3m` | `2.0h` | `1.9h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:09` | `hansolefsen2012 -> tomreyn` | `4m` | `2.0h` | `1.9h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:13` | `hansolefsen2012 -> tomreyn` | `4m` | `2.0h` | `1.9h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T10:06` | `hansolefsen2012 -> tomreyn` | `4m` | `2.0h` | `1.9h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:12` | `tomreyn -> hansolefsen2012` | `5m` | `2.0h` | `1.9h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:30` | `tomreyn -> hansolefsen2012` | `10m` | `2.0h` | `1.8h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |
| `2026-02-04T09:17` | `tomreyn -> hansolefsen2012` | `13m` | `2.0h` | `1.8h` | `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html` |

## Sample Contexts

### 2026-02-06T16:26 - SuperLag to lotuspsychje

- Human next-turn delay: `5.9h`
- Model prediction: `37s`
- Source: `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html`

```text
2026-02-06T16:07 SuperLag: I'm running Ubuntu 25.10, and neither Flameshot nor Shutter can take screenshots. I get messages like https://imgur.com/a/trying-to-take-screenshot-with-shutter-on-ubuntu-25-10-Tbu7DsE
2026-02-06T16:10 lotuspsychje: SuperLag: 25.10 doesnt have xorg anymore flameshot should be wayland capable
2026-02-06T16:12 SuperLag: "Flameshot error: unable to capture screen"
2026-02-06T16:14 lotuspsychje: SuperLag: you're right, just tested flameshot on 26.04 wayland, errors too SuperLag: bug #2138139 you know howto affect yourself to the bug SuperLag?
2026-02-06T16:18 lotuspsychje: SuperLag: try launching from terminal, works there for me
2026-02-06T16:21 SuperLag: launch it from the terminal? you're saying that works around the bug?
2026-02-06T16:21 lotuspsychje: yeah for my case
2026-02-06T16:26 SuperLag: Hmm... looks like it's a Flatpak. It's not on $PATH
```

### 2026-02-04T18:41 - JanC to osse

- Human next-turn delay: `5.2h`
- Model prediction: `25s`
- Source: `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html`

```text
2026-02-04T18:22 osse: I tried to do a distro upgrade and it failed. It seems to be partially upgraded but totally borked. Don't have sudo, apt doesn't work and dpkg --reconfigure -a didn't. Luckily I had sudo-rs so I managed to manually download the libapt-pkg7.0 deb, which made apt work, which let me upgrade python3 which in turn has made dpkg --reconfigure actually do something \o/
2026-02-04T18:27 osse: lotuspsychje: 24.04 to 25.10 Did the same thing at work and that went without a hitch.
2026-02-04T18:34 osse: lotuspsychje: hmm, that could be. all the more motivation to make it work :P dpkg --reconfigure -a seems to have worked. so I will attempt a reboot now
2026-02-04T18:35 JanC: how did it fail? I assume you used APT to upgrade, not the official upgrader?
2026-02-04T18:36 osse: JanC: I don't know. The error messages were quite vague. I'll see if I can find some logs JanC: I just changed the settings for what kind of Ubuntu version I should be propted for in the "Software and Updates" thing from only LTS to any, then clicked "Upgrade" when I was offered to
2026-02-04T18:39 JanC: oh, I thought that it would not allow upgrading except to 24.10 then
2026-02-04T18:41 osse: Heh, rebooting worked. But I get something similar to the default gnome-shell (but with the ubuntu background) and / is a "read-only file system" so I can't do much
2026-02-04T18:41 JanC: I wonder why read-only / you can try to re-mount / as rw
```

### 2026-02-06T13:05 - tomreyn to ajorj

- Human next-turn delay: `4.8h`
- Model prediction: `26s`
- Source: `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html`

```text
2026-02-06T13:05 ajorj: Driver versions the same in live env and after installation?
2026-02-06T13:05 tomreyn: you could try booting without the "splash" option in /etc/default/grub (and run sudo update-grub afterwards)
```

### 2026-02-06T13:07 - tomreyn to TomyWork

- Human next-turn delay: `4.8h`
- Model prediction: `40s`
- Source: `https://irclogs.ubuntu.com/2026/02/06/%23ubuntu.html`

```text
2026-02-06T13:02 TomyWork: my coworker's new kubuntu 24.04 doesn't work with the monitors connected to his docking station. what's odd is that they work when booting from a live iso. we think the difference is the pre-boot environment that asks for the password and mounts the encrypted disks. any idea how to address this issue? we already tried reconnecting the docking station after booting, but that doesn't fix the monitor issue in any of our attempts
2026-02-06T13:04 TomyWork: daisy chaining is still a rare feature. i doubt those monitors support that
2026-02-06T13:05 TomyWork: like i said they all work in a live distro
2026-02-06T13:05 TomyWork: kubuntu 24.04 live iso, booted via ventoy, to be precise
2026-02-06T13:05 tomreyn: you could try booting without the "splash" option in /etc/default/grub (and run sudo update-grub afterwards)
2026-02-06T13:06 TomyWork: ajorj, what's a "driver", exactly, on linux? :) he says it's intel hd, no discrete nvidia or amd chip present tomreyn, worth a try, thanks
2026-02-06T13:07 tomreyn: i'm not sure it helps, it's just a guess
```

### 2026-02-04T14:17 - TomyWork to leftyfb

- Human next-turn delay: `2.0h`
- Model prediction: `36s`
- Source: `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html`

```text
2026-02-04T10:17 TomyWork: I fell into the lxc trap again: Feb 04 11:10:22 prod-git lxd.daemon[1073551]: => First LXD execution on this system noticed lxc was installed, ran "lxc list" to see what was running and whether I can get rid of it, but that acttually installed lxc is removing the lxd snap enough to undo that?
2026-02-04T11:00 TomyWork: pip install linux
2026-02-04T14:10 leftyfb: TomyWork: sudo apt install remove lxd-installer
2026-02-04T14:13 leftyfb: damnit, I need to not post on IRC before 10am TomyWork: sudo apt remove lxd-installer
2026-02-04T14:17 TomyWork: leftyfb, and that'll take care of undoing all the stuff the installer did, like loading the zfs module for some reason?
```

### 2026-02-04T09:23 - hansolefsen2012 to lazysundaydreams

- Human next-turn delay: `3m`
- Model prediction: `2.0h`
- Source: `https://irclogs.ubuntu.com/2026/02/04/%23ubuntu.html`

```text
2026-02-04T08:54 hansolefsen2012: Hello, ananke , can you the link to nginx via apache as load balancer? Found solution on stackoverflow but stack of memory is not correct...
2026-02-04T09:00 hansolefsen2012: I'm sorry that I'm hijacking the topic but what is ncdu - "Norton commander disk usage?
2026-02-04T09:01 hansolefsen2012: I've got it... Fuck, there are a lot of features had changed in Ubuntu, so have to study, a month after there would be a lot of snippets in stored folder...
2026-02-04T09:07 hansolefsen2012: Did you compiled the kernel? There would be the opportunity to load much faster if remove modules like for graphic card and new pci hot plug, scsi etc
2026-02-04T09:09 hansolefsen2012: 2 hours of configuring linux kernel modules and booting in 25 seconds I mean with gnome, men Never used /home/user/smth_skel
2026-02-04T09:13 hansolefsen2012: But they didn't die. Please do take a seat and enjoy bloody rayan as I guess was the best horror action in 2003-2004, not on theme...
2026-02-04T09:17 hansolefsen2012: ok
2026-02-04T09:20 hansolefsen2012: In the future I would rent the virtual servers with Ubuntu for clustering projects, so, exp users write you offers for hour rate. I'm too old for studying🦊🦊😍☘️😎
2026-02-04T09:22 lazysundaydreams: hansolefsen2012 params?
2026-02-04T09:23 hansolefsen2012: In hour village a lot of garbage cleaners and even simple people work for 8-9 dollars per hours that's why potato costs 27 uahs per kilo. #8-9 #12
```

