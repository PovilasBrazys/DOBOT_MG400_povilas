-- Version: Lua 5.3.5
-- Dobot MG400 TCP server that receives Y coordinates and moves the robot accordingly

local ip = "192.168.2.6"
local port = 8000
local err = 0
local socket = 0

::create_server::
err, socket = TCPCreate(true, ip, port)
if err ~= 0 then
    print("Failed to create socket, re-connecting")
    Sleep(1000)
    goto create_server
end

err = TCPStart(socket, 0)
if err ~= 0 then
    print("Failed to start server, re-connecting")
    TCPDestroy(socket)
    Sleep(1000)
    goto create_server
end

print("TCP Server running on " .. ip .. ":" .. port)

while true do
    err, buf = TCPRead(socket, 0, "string")
    if err ~= 0 then
        print("Failed to read data, re-connecting")
        TCPDestroy(socket)
        Sleep(1000)
        goto create_server
    end

    local data = buf.buf
    print("Received:", data)

    -- Attempt to parse a message like "Y=-23.5"
    local y_value = tonumber(string.match(data, "Y=([%-%d%.]+)"))

    if y_value then
        print("Parsed Y =", y_value)

        -- Clamp Y to safety range
        if y_value > 50 then y_value = 50 end
        if y_value < -50 then y_value = -50 end

        -- Perform motion to X=350, Y=y_value, Z=0, R=0
        local target_pose = {coordinate = {350, y_value, 0, 0}, tool = 0, user = 0}
        MovJ(target_pose)
        TCPWrite(socket, "OK")
    else
        print("Invalid command received")
        TCPWrite(socket, "ERR")
    end

    Sleep(50)
end
