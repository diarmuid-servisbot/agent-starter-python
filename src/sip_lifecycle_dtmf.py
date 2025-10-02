import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit import rtc
from livekit import api
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero, cartesia

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram

from typing import Annotated, Optional
from pydantic import Field
from dataclasses import dataclass
import time

# If ENV_FILE is set, load a specific .env file. Otherwise, assume env vars are set directly.
# This is the standard practice for server environments like Coolify.
env_file = os.getenv("ENV_FILE")
if env_file:
    print(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

# Configure file logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('agent_debug.log', maxBytes=5242880, backupCount=3)  # 5MB
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# Configure console logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Set up main logger
logger = logging.getLogger("sip-lifecycle-agent")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Enable debug logging for OpenAI and LLM calls - log to file
llm_logger = logging.getLogger("livekit.agents.llm")
llm_logger.setLevel(logging.DEBUG)
llm_logger.addHandler(file_handler)

openai_logger = logging.getLogger("livekit.plugins.openai")
openai_logger.setLevel(logging.DEBUG)
openai_logger.addHandler(file_handler)

# Also log function tool calls
function_logger = logging.getLogger("livekit.agents.llm.function_tool")
function_logger.setLevel(logging.DEBUG)
function_logger.addHandler(file_handler)


@dataclass
class UserData:
    """Store user data for the navigator agent."""
    ctx: JobContext
    last_dtmf_press: float = 0

RunContext_T = RunContext[UserData]

class DeepgramSTTWithVAD(deepgram.STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _connect(self, *args, **kwargs):
        # Call parent connect
        conn = await super()._connect(*args, **kwargs)

        # Inject Deepgram raw options
        # (These are passed in the initial streaming request)
        # See: https://developers.deepgram.com/docs/streaming
        conn._dg_request["endpointing"] = True
        conn._dg_request["vad_turnoff"] = 500  # ms after silence

        return conn


class LoggingOpenAI(openai.LLM):
    def __init__(self, *args, **kwargs):
        # Override with environment variables if provided
        if 'model' not in kwargs:
            kwargs['model'] = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        # Set custom base URL if provided
        if 'base_url' not in kwargs and os.getenv('OPENAI_BASE_URL'):
            kwargs['base_url'] = os.getenv('OPENAI_BASE_URL')
        
        # Set custom API key if provided
        if 'api_key' not in kwargs and os.getenv('OPENAI_API_KEY'):
            kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
            
        super().__init__(*args, **kwargs)
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("llm_logs")
        self.logs_dir.mkdir(exist_ok=True)

    async def _make_request(self, messages, **kwargs):
        timestamp = datetime.now()
        
        # Create timestamped filename
        filename = f"llm_call_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.json"
        log_file = self.logs_dir / filename
        
        # Prepare log data
        log_data = {
            "timestamp": timestamp.isoformat(),
            "request": {
                "messages": messages,
                "kwargs": kwargs
            }
        }
        
        # Log input to console
        print(f"🔵 OpenAI Request [{timestamp.strftime('%H:%M:%S')}]:")
        for m in messages:
            content = m.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  {m['role']}: {content}")

        try:
            # Call the parent method
            response = await super()._make_request(messages, **kwargs)
            
            # Add response to log data
            log_data["response"] = response
            log_data["status"] = "success"
            
            # Log output to console
            print(f"🟢 OpenAI Response [{timestamp.strftime('%H:%M:%S')}]: Success")
            
        except Exception as e:
            # Log error
            log_data["error"] = str(e)
            log_data["status"] = "error"
            print(f"🔴 OpenAI Error [{timestamp.strftime('%H:%M:%S')}]: {e}")
            raise
        
        finally:
            # Write to timestamped file
            try:
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to write LLM log to {log_file}: {e}")

        return response

class SIPLifecycleAgent(Agent):
    def __init__(self, job_context=None) -> None:
        self.job_context = job_context
        
        # Load instructions from file
        try:
            with open("escrow.txt", "r") as f:
                instructions = f.read()
        except FileNotFoundError:
            logger.error("instructions.txt not found, using default instructions")
            instructions = "You are an AI assistant specializing in mortgage escrow inquiries."
        
        stt = DeepgramSTTWithVAD(
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
            interim_results=True
        )

        llm = LoggingOpenAI()  # Will use environment variables or defaults

        userdata = UserData(ctx=self.job_context)

        super().__init__(
            instructions=instructions,
            turn_detection=EnglishModel(),
            stt=stt,
            llm=llm,
            tts=deepgram.TTS( model="aura-asteria-en",),
            min_endpointing_delay=0.75
        )

    @function_tool
    async def add_sip_participant(self, context: RunContext, phone_number: str):
        """
        Add a SIP participant to the current call.
        
        Args:
            context: The call context
            phone_number: The phone number to call
        """
        if not self.job_context:
            logger.error("No job context available")
            await self.session.say("I'm sorry, I can't add participants at this time.")
            return None, "Failed to add SIP participant: No job context available"
            
        room_name = self.job_context.room.name
        
        identity = f"sip_{uuid.uuid4().hex[:8]}"
        
        sip_trunk_id = os.environ.get('SIP_TRUNK_ID')
        
        logger.info(f"Adding SIP participant with phone number {phone_number} to room {room_name}")
        
        try:
            response = await self.job_context.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    room_name=room_name,
                    participant_identity=identity,
                    participant_name=f"SIP Participant {phone_number}",
                    krisp_enabled=True
                )
            )
            
            logger.info(f"Successfully added SIP participant: {response}")
            return None, f"Added SIP participant {phone_number} to the call."
            
        except Exception as e:
            logger.error(f"Error adding SIP participant: {e}")
            await self.session.say(f"I'm sorry, I couldn't add {phone_number} to the call.")
            return None, f"Failed to add SIP participant: {e}"

    @function_tool
    async def end_call(self, context: RunContext):
        """
        End the current call by deleting the room.
        """
        if not self.job_context:
            logger.error("No job context available")
            await self.session.say("I'm sorry, I can't end the call at this time.")
            return None, "Failed to end call: No job context available"
            
        room_name = self.job_context.room.name
        logger.info(f"Ending call by deleting room {room_name}")
        
        try:
            await context.session.generate_reply(
                instructions="Thank you for your time. I'll be ending this call now. Goodbye!"
            )
            await self.job_context.api.room.delete_room(
                api.DeleteRoomRequest(room=room_name)
            )
            
            logger.info(f"Successfully deleted room {room_name}")
            return None, "Call ended successfully."
            
        except Exception as e:
            logger.error(f"Error ending call: {e}")
            return None, f"Failed to end call: {e}"

    @function_tool
    async def log_participants(self, context: RunContext):
        """
        Log all participants in the current room.
        """
        if not self.job_context:
            logger.error("No job context available")
            await self.session.say("I'm sorry, I can't list participants at this time.")
            return None, "Failed to list participants: No job context available"
            
        room_name = self.job_context.room.name
        logger.info(f"Logging participants in room {room_name}")
        
        try:
            response = await self.job_context.api.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            
            participants = response.participants
            participant_info = []
            
            for p in participants:
                participant_info.append({
                    "identity": p.identity,
                    "name": p.name,
                    "state": p.state,
                    "is_publisher": p.is_publisher
                })
            
            logger.info(f"Participants in room {room_name}: {participant_info}")
            
            await self.session.say(f"There are {len(participants)} participants in this call.")
            
            return None, f"Listed {len(participants)} participants in the room."
            
        except Exception as e:
            logger.error(f"Error listing participants: {e}")
            return None, f"Failed to list participants: {e}"

    @function_tool
    async def play_joke(self, context: RunContext) -> dict:
        """
        Function to read and return a random joke from the jokes file.
        Returns:
            dict: Response containing the joke, or an error status.
        """
        try:
            with open("jokes/mortgage_jokes.txt", "r") as f:
                # Split on double newlines to separate jokes
                jokes = [j.strip() for j in f.read().split("\n\n") if j.strip()]
                if not jokes:
                    logger.warning("No jokes found in jokes/mortgage_jokes.txt")
                    await self.session.say("Sorry, I'm all out of jokes right now!")
                    return {
                        "status": "error",
                        "message": "Sorry, I'm all out of jokes right now!",
                        "joke": None
                    }
                joke = random.choice(jokes)
                logger.info(f"Selected joke: {joke}")
                await self.session.say(f"Here's a joke for you! {joke}")
                return {
                    "status": "success",
                    "message": "Here's a joke for you!",
                    "joke": joke
                }
        except FileNotFoundError:
            logger.error("jokes/mortgage_jokes.txt not found.")
            await self.session.say("Sorry, I can't seem to find my joke book right now.")
            return {
                "status": "error",
                "message": "Sorry, I can't seem to find my joke book right now.",
                "joke": None
            }
        except Exception as e:
            logger.error(f"Error reading joke: {e}")
            await self.session.say("Sorry, I had a little trouble remembering a joke.")
            return {
                "status": "error",
                "message": "Sorry, I had a little trouble remembering a joke.",
                "joke": None
            }


    @function_tool()
    async def play_dtmf(
        self,
        code: Annotated[int, Field(description="The DTMF code to send to the phone number for the current step.")],
        context: RunContext_T
    ) -> None:
        """Called when you need to send a DTMF code to the phone number for the current step."""
        current_time = time.time()
        
        # Check if enough time has passed since last press (3 second cooldown)
        if current_time - context.userdata.last_dtmf_press < 3:
            logger.info("DTMF code rejected due to cooldown")
            return None
            
        logger.info(f"Sending DTMF code {code} to the phone number for the current step.")
        context.userdata.last_dtmf_press = current_time
        
        room = context.userdata.ctx.room

        await room.local_participant.publish_dtmf(
            code=code,
            digit=str(code)
        )
        await room.local_participant.publish_data(
            f"{code}",
            topic="dtmf_code"
        )
        return None


    async def play_dtmf_yy(self, context: RunContext, digits: str) -> dict:
        """
        Play Dual-Tone Multi-Frequency (DTMF) digits into the current active call
        through the LiveKit session. This simulates pressing keys on a phone
        keypad and is typically used to interact with IVR (Interactive Voice
        Response) systems, voicemail menus, or conference bridges that expect
        keypad input.

        Use this function when:
        - The caller requests to "press" or "enter" digits, such as
            "press 1 for sales" or "enter my account number."
        - An IVR system prompts for input using numeric or special characters
            (0–9, *, #, A–D).
        - You need to navigate phone menus or send confirmation tones.

        Args:
            context (RunContext): The LiveKit agent run context (provided automatically).
            digits (str): A sequence of one or more DTMF characters.
                - Acceptable values: 0–9, *, #, A–D (case-insensitive).
                - Example: "123#", "*0", "45A".

        Returns:
            dict: A dictionary containing the status of the operation.
                Example on success:
                {
                    "status": "success",
                    "digits": "123#",
                    "message": "Played DTMF digits 123# into the call."
                }

                Example on error:
                {
                    "status": "error",
                    "digits": "123#",
                    "message": "Failed to play DTMF: <error details>"
                }

        Notes for the LLM agent:
        - Call this function ONLY when DTMF tones are explicitly required
            (e.g., interacting with automated phone menus).
        - Do NOT use this function for natural conversation with the caller.
        - Always provide the full sequence of digits in the "digits" argument
            as a string, without spaces.
        """
        try:
            if not digits or not isinstance(digits, str):
                raise ValueError("Digits must be a non-empty string")

            logger.info(f"Sending DTMF digits: {digits}")

            mapping = {
                '0': 0, '1': 1, '2': 2, '3': 3,
                '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9,
                '*': 10, '#': 11,
                'A': 12, 'B': 13, 'C': 14, 'D': 15,
            }
            print(dir(self.session))
            print('x'*50)
            local_participant = self.session.participant

            for d in digits:
                if d.upper() in mapping:
                    await local_participant.publish_dtmf(
                        code=mapping[d.upper()],
                        digit=d
                    )
                    logger.debug(f"Sent DTMF {d}")
                else:
                    logger.warning(f"Invalid DTMF char skipped: {d}")

            return {
                "status": "success",
                "digits": digits,
                "message": f"Played DTMF digits {digits} into the call."
            }

        except Exception as e:
            logger.error(f"Error sending DTMF: {e}")
            return {
                "status": "error",
                "message": f"Failed to play DTMF: {e}",
                "digits": digits
            }


    async def play_dtmfxxx(self, context: RunContext, digits: str) -> dict:
        """
        Function to play a sequence of DTMF digits into the current call.
        Args:
            context (RunContext): LiveKit agent run context
            digits (str): Sequence of digits to play, e.g. "123#*"
        Returns:
            dict: status and echo of digits played
        """
        try:
            if not digits or not isinstance(digits, str):
                raise ValueError("Digits must be a non-empty string")

            logger.info(f"Playing DTMF into call: {digits}")
            
            # Get the local participant from the agent's room
            local_participant = self.session.room.local_participant
            
            # Send each digit individually
            for digit in digits:
                if digit.isdigit():
                    code = int(digit)
                    await local_participant.publish_dtmf(code=code, digit=digit)
                elif digit == '*':
                    await local_participant.publish_dtmf(code=10, digit='*')
                elif digit == '#':
                    await local_participant.publish_dtmf(code=11, digit='#')
                else:
                    logger.warning(f"Skipping invalid DTMF character: {digit}")

            return {
                "status": "success",
                "message": f"Played DTMF digits {digits} into the call.",
                "digits": digits
            }
        except Exception as e:
            logger.error(f"Error sending DTMF: {e}")
            return {
                "status": "error",
                "message": f"Failed to play DTMF: {e}",
                "digits": digits
            }
                        
    async def on_enter(self):
        self.session.generate_reply()

async def entrypoint(ctx: JobContext):
    session = AgentSession()
    agent = SIPLifecycleAgent(job_context=ctx)

    await session.start(
        agent=agent,
        room=ctx.room
    )

    def on_participant_connected_handler(participant: rtc.RemoteParticipant):
        asyncio.create_task(async_on_participant_connected(participant))

    def on_participant_attributes_changed_handler(changed_attributes: dict, participant: rtc.Participant):
        asyncio.create_task(async_on_participant_attributes_changed(changed_attributes, participant))

    # New: DTMF handler
    def on_dtmf_handler(evt):
        asyncio.create_task(async_on_dtmf(evt))

    async def async_on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"New participant connected: {participant.identity}")

        # Check if this is a SIP participant and log call status
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.info(f"SIP participant connected: {participant.identity}")

            # Get the task from attributes
            task = participant._info.attributes.get("task")
            logger.info(f"task: {task}")

            userdata = UserData(ctx=ctx, task=task)

            # Log SIP attributes
            if participant.attributes:
                call_id = participant.attributes.get('sip.callID', 'Unknown')
                call_status = participant.attributes.get('sip.callStatus', 'Unknown')
                phone_number = participant.attributes.get('sip.phoneNumber', 'Unknown')
                trunk_id = participant.attributes.get('sip.trunkID', 'Unknown')
                trunk_phone = participant.attributes.get('sip.trunkPhoneNumber', 'Unknown')

                logger.info(f"SIP Call ID: {call_id}")
                logger.info(f"SIP Call Status: {call_status}")
                logger.info(f"SIP Phone Number: {phone_number}")
                logger.info(f"SIP Trunk ID: {trunk_id}")
                logger.info(f"SIP Trunk Phone Number: {trunk_phone}")

                # Log specific call status information
                if call_status == 'active':
                    logger.info("Call is active and connected")
                elif call_status == 'automation':
                    logger.info("Call is connected and dialing DTMF numbers")
                elif call_status == 'dialing':
                    logger.info("Call is dialing and waiting to be picked up")
                elif call_status == 'hangup':
                    logger.info("Call has been ended by a participant")
                elif call_status == 'ringing':
                    logger.info("Inbound call is ringing for the caller")

        await agent.session.say(f"Welcome, {participant.name or participant.identity}! I can help you add a participant to this call or end the call.")

    async def async_on_participant_attributes_changed(changed_attributes: dict, participant: rtc.Participant):
        logger.info(f"Participant {participant.identity} attributes changed: {changed_attributes}")

        # Check if this is a SIP participant and if call status has changed
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            # Check if sip.callStatus is in the changed attributes
            if 'sip.callStatus' in changed_attributes:
                call_status = changed_attributes['sip.callStatus']
                logger.info(f"SIP Call Status updated: {call_status}")

                # Log specific call status information
                if call_status == 'active':
                    logger.info("Call is now active and connected")
                elif call_status == 'automation':
                    logger.info("Call is now connected and dialing DTMF numbers")
                elif call_status == 'dialing':
                    logger.info("Call is now dialing and waiting to be picked up")
                elif call_status == 'hangup':
                    logger.info("Call has been ended by a participant")
                elif call_status == 'ringing':
                    logger.info("Inbound call is now ringing for the caller")

    # New async DTMF handler logic
    async def async_on_dtmf(evt):
        logger.info(f"DTMF event received: {evt}")
        try:
            participant_id = getattr(evt, "participant", {}).identity if hasattr(evt, "participant") else evt.get("participant", {}).get("identity")
            digit = getattr(evt, "digit", None) or evt.get("digit")
            logger.info(f"DTMF from {participant_id}: {digit}")
            await agent.session.say(f"I heard you press {digit}")
        except Exception as e:
            logger.error(f"Error processing DTMF: {e}")

    ctx.room.on("participant_connected", on_participant_connected_handler)
    ctx.room.on("participant_attributes_changed", on_participant_attributes_changed_handler)

    # Register the DTMF event
    ctx.room.on("sip_dtmf_received", on_dtmf_handler)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))