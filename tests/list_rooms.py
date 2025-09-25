import asyncio
from livekit.api import LiveKitAPI
from livekit.api.room_service import ListRoomsRequest, ListParticipantsRequest

async def main():
    lkapi = LiveKitAPI(
        url="https://livekit.preview.servismatrix.com",
        api_key="devkey",
        api_secret="secret",
    )

    # List rooms
    resp = await lkapi.room.list_rooms(ListRoomsRequest())
    if not resp.rooms:
        print("No active rooms.")
    else:
        print("=== Active Rooms ===")
        for room in resp.rooms:
            print(f"- {room.name}, max participants: {room.max_participants}")

        # List participants in the first room
        room_name = resp.rooms[0].name
        part_resp = await lkapi.room.list_participants(
            ListParticipantsRequest(room=room_name)
        )
        print(f"\n=== Participants in {room_name} ===")
        for p in part_resp.participants:
            print(f"- {p.identity} (state: {p.state}, metadata: {p.metadata})")

    await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(main())
