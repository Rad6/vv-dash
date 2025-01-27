## DRV - DRaco encoded Video format version 1.0.0

```text
DRV Version: - 1.0.0
    format version major (1 byte)
    format version minor (1 byte)
    format version patch (1 byte)

Header:
    Header Encoding (4 bytes) - ASCII("JSON")
    Header Size (4 bytes)
    Head Payload
Frames:
    Frame 1 data
    Frame 2 data
    ...
```


## Header payload format (JSON)

Header will be ASCII encoded JSON.

Frame display time: `pts/timescale`
```json
{
    "timescale": 100,
    "frames": [
        {"offset": 0, "size": 10032, "pts": 0},
        {"offset": 1234, "pts": 10},
        {"offset": 32423, "pts": 20},
        {"offset": 23343, "pts": 30}
    ]
}
```