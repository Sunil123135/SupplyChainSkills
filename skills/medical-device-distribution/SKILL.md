---
name: medical-device-distribution
description: "When the user wants to optimize medical device distribution, manage device traceability, handle consignment inventory, or ensure regulatory compliance for medical devices. Also use when the user mentions \"medical device logistics,\" \"UDI compliance,\" \"device traceability,\" \"consignment management,\" \"implant tracking,\" \"loaner sets,\" \"FDA compliance,\" \"sterile device distribution,\" \"recall management,\" or \"GS1 standards.\" For hospital internal logistics, see hospital-logistics. For pharmaceutical distribution, see pharmacy-supply-chain."
---

# Medical Device Distribution

You are an expert in medical device distribution and supply chain management. Your goal is to ensure compliant, efficient distribution of medical devices while maintaining traceability, managing regulatory requirements, and optimizing inventory costs.

## Initial Assessment

Before optimizing medical device distribution, understand:

1. **Device Categories**
   - Device classifications? (Class I, II, III)
   - Product types? (implants, capital equipment, disposables, loaner sets)
   - High-value vs. high-volume products?
   - Sterile vs. non-sterile requirements?

2. **Regulatory Landscape**
   - FDA registration status?
   - UDI compliance requirements?
   - QMS (Quality Management System) in place?
   - International distribution? (CE Mark, other countries)

3. **Distribution Network**
   - Direct distribution vs. through distributors?
   - Number of distribution centers?
   - Customer types? (hospitals, clinics, surgery centers)
   - Geographic coverage?

4. **Current Challenges**
   - Traceability gaps?
   - Recall preparedness?
   - Consignment inventory management?
   - Expiry/obsolescence issues?
   - Compliance violations or warning letters?

---

## Medical Device Distribution Framework

### Device Classification & Requirements

**FDA Device Classes:**

**Class I (Low Risk):**
- Examples: Bandages, examination gloves, handheld instruments
- Requirements: General controls, most exempt from 510(k)
- Distribution: Standard supply chain, minimal special handling

**Class II (Moderate Risk):**
- Examples: Powered wheelchairs, infusion pumps, surgical drapes
- Requirements: General + special controls, 510(k) clearance usually required
- Distribution: Enhanced traceability, often UDI required

**Class III (High Risk):**
- Examples: Pacemakers, heart valves, implantable defibrillators
- Requirements: Premarket approval (PMA), strictest controls
- Distribution: Full traceability, serialization, consignment common

### Unique Device Identification (UDI) Compliance

**UDI Requirements:**
- Unique identifier on device label and packaging
- Direct marking on implantable devices
- Registration in FDA's GUDID (Global Unique Device Identification Database)
- Tracking through distribution chain

**UDI Structure:**
```
UDI = Device Identifier (DI) + Production Identifier (PI)

DI: Identifies the specific device (model, version)
PI: Identifies the production unit (lot, serial number, expiry, manufacture date)
```

**Implementation:**

```python
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class UDI:
    """
    Unique Device Identification structure
    """
    device_identifier: str  # DI - identifies device model
    lot_number: Optional[str] = None
    serial_number: Optional[str] = None
    manufacturing_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    donation_id: Optional[str] = None  # For blood/tissue

    def to_gs1_string(self):
        """Convert to GS1 format"""
        udi_string = f"(01){self.device_identifier}"

        if self.lot_number:
            udi_string += f"(10){self.lot_number}"

        if self.serial_number:
            udi_string += f"(21){self.serial_number}"

        if self.expiration_date:
            exp_date = self.expiration_date.strftime("%y%m%d")
            udi_string += f"(17){exp_date}"

        if self.manufacturing_date:
            mfg_date = self.manufacturing_date.strftime("%y%m%d")
            udi_string += f"(11){mfg_date}"

        return udi_string

    @staticmethod
    def parse_gs1(gs1_string):
        """Parse GS1 UDI string"""
        # Application Identifiers (AI)
        patterns = {
            'device_identifier': r'\(01\)(\d{14})',
            'lot_number': r'\(10\)([A-Za-z0-9]+)',
            'serial_number': r'\(21\)([A-Za-z0-9]+)',
            'expiration_date': r'\(17\)(\d{6})',
            'manufacturing_date': r'\(11\)(\d{6})'
        }

        udi_data = {}

        for field, pattern in patterns.items():
            match = re.search(pattern, gs1_string)
            if match:
                value = match.group(1)
                if 'date' in field:
                    # Convert YYMMDD to datetime
                    udi_data[field] = datetime.strptime(value, "%y%m%d")
                else:
                    udi_data[field] = value

        return UDI(**udi_data)

# Example usage
device_udi = UDI(
    device_identifier="10884521123456",
    lot_number="LOT2024A",
    serial_number="SN123456789",
    expiration_date=datetime(2027, 12, 31)
)

gs1_string = device_udi.to_gs1_string()
print(f"GS1 UDI: {gs1_string}")

# Parse back
parsed_udi = UDI.parse_gs1(gs1_string)
print(f"Parsed Serial: {parsed_udi.serial_number}")
print(f"Parsed Expiry: {parsed_udi.expiration_date}")
```

---

## Traceability & Serialization

### End-to-End Traceability System

```python
import pandas as pd
from datetime import datetime
from enum import Enum

class EventType(Enum):
    MANUFACTURED = "manufactured"
    SHIPPED = "shipped"
    RECEIVED = "received"
    IMPLANTED = "implanted"
    RETURNED = "returned"
    RECALLED = "recalled"

class TraceabilitySystem:
    """
    Track medical devices through supply chain
    """

    def __init__(self):
        self.events = []

    def record_event(self, event_type, device_id, serial_number,
                     location, operator, timestamp=None, metadata=None):
        """
        Record a traceability event

        Parameters:
        - event_type: Type of event (EventType enum)
        - device_id: Device identifier (DI)
        - serial_number: Unique serial number
        - location: Where event occurred
        - operator: Who performed action
        - timestamp: When it occurred (defaults to now)
        - metadata: Additional data (patient ID, lot, etc.)
        """

        if timestamp is None:
            timestamp = datetime.now()

        event = {
            'event_type': event_type.value,
            'device_id': device_id,
            'serial_number': serial_number,
            'location': location,
            'operator': operator,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        self.events.append(event)

        return event

    def trace_device_history(self, serial_number):
        """
        Get complete history for a device by serial number
        """

        device_events = [
            e for e in self.events
            if e['serial_number'] == serial_number
        ]

        # Sort by timestamp
        device_events.sort(key=lambda x: x['timestamp'])

        return pd.DataFrame(device_events)

    def find_devices_by_lot(self, lot_number):
        """
        Find all devices from a specific lot (for recalls)
        """

        devices = [
            e for e in self.events
            if e.get('metadata', {}).get('lot_number') == lot_number
        ]

        # Get unique serial numbers
        serial_numbers = list(set(e['serial_number'] for e in devices))

        return serial_numbers

    def implanted_devices_report(self, start_date, end_date):
        """
        Report of devices implanted in date range
        """

        implanted = [
            e for e in self.events
            if e['event_type'] == EventType.IMPLANTED.value
            and start_date <= e['timestamp'] <= end_date
        ]

        return pd.DataFrame(implanted)

    def audit_trail(self, device_id=None, location=None, date_range=None):
        """
        Generate audit trail for compliance
        """

        filtered_events = self.events

        if device_id:
            filtered_events = [e for e in filtered_events if e['device_id'] == device_id]

        if location:
            filtered_events = [e for e in filtered_events if e['location'] == location]

        if date_range:
            start, end = date_range
            filtered_events = [
                e for e in filtered_events
                if start <= e['timestamp'] <= end
            ]

        return pd.DataFrame(filtered_events)

# Example usage
traceability = TraceabilitySystem()

# Manufacturing event
traceability.record_event(
    EventType.MANUFACTURED,
    device_id="10884521123456",
    serial_number="SN123456789",
    location="Manufacturing Plant A",
    operator="Production Line 3",
    metadata={'lot_number': 'LOT2024A', 'manufacturing_date': '2024-01-15'}
)

# Shipment to distributor
traceability.record_event(
    EventType.SHIPPED,
    device_id="10884521123456",
    serial_number="SN123456789",
    location="Distribution Center East",
    operator="Warehouse Operator J.Smith",
    metadata={'tracking_number': 'TRACK123', 'carrier': 'FedEx'}
)

# Receipt at hospital
traceability.record_event(
    EventType.RECEIVED,
    device_id="10884521123456",
    serial_number="SN123456789",
    location="Memorial Hospital",
    operator="Materials Manager",
    metadata={'purchase_order': 'PO-2024-5678'}
)

# Implantation
traceability.record_event(
    EventType.IMPLANTED,
    device_id="10884521123456",
    serial_number="SN123456789",
    location="Memorial Hospital OR-3",
    operator="Dr. Johnson",
    metadata={
        'patient_id': 'PT-987654',  # PHI - must be secured
        'procedure_code': 'CPT-33206',
        'surgeon': 'Dr. Johnson'
    }
)

# Get device history
history = traceability.trace_device_history("SN123456789")
print("Device History:")
print(history[['timestamp', 'event_type', 'location', 'operator']])
```

---

## Consignment Inventory Management

### Consignment Optimization Model

**Consignment Characteristics:**
- Vendor owns inventory until use
- Located at customer site (hospital)
- Hospital pays only when consumed
- Vendor responsible for replenishment
- Common for high-value implants

```python
import numpy as np
import pandas as pd

class ConsignmentInventoryManager:
    """
    Manage consignment inventory for medical devices
    """

    def __init__(self, location_name, par_level_policy=None):
        self.location_name = location_name
        self.inventory = {}  # {serial_number: device_info}
        self.par_levels = par_level_policy or {}
        self.usage_history = []

    def add_device(self, device_id, serial_number, cost, vendor, metadata=None):
        """Add device to consignment inventory"""

        self.inventory[serial_number] = {
            'device_id': device_id,
            'serial_number': serial_number,
            'cost': cost,
            'vendor': vendor,
            'status': 'available',
            'received_date': datetime.now(),
            'metadata': metadata or {}
        }

    def consume_device(self, serial_number, patient_id, procedure_info):
        """
        Record device consumption (trigger payment to vendor)
        """

        if serial_number not in self.inventory:
            raise ValueError(f"Device {serial_number} not in consignment inventory")

        device = self.inventory[serial_number]

        if device['status'] != 'available':
            raise ValueError(f"Device {serial_number} is not available (status: {device['status']})")

        # Record usage
        usage_record = {
            'serial_number': serial_number,
            'device_id': device['device_id'],
            'vendor': device['vendor'],
            'cost': device['cost'],
            'consumption_date': datetime.now(),
            'patient_id': patient_id,
            'procedure_info': procedure_info,
            'days_in_consignment': (datetime.now() - device['received_date']).days
        }

        self.usage_history.append(usage_record)

        # Update device status
        device['status'] = 'consumed'
        device['consumption_date'] = datetime.now()

        return usage_record

    def check_par_levels(self):
        """
        Check if consignment inventory meets PAR levels
        Trigger vendor replenishment if needed
        """

        replenishment_needed = []

        for device_id, par_level in self.par_levels.items():
            # Count available devices
            available_count = sum(
                1 for device in self.inventory.values()
                if device['device_id'] == device_id and device['status'] == 'available'
            )

            if available_count < par_level:
                shortage = par_level - available_count

                replenishment_needed.append({
                    'device_id': device_id,
                    'current_qty': available_count,
                    'par_level': par_level,
                    'replenishment_qty': shortage
                })

        return replenishment_needed

    def vendor_reconciliation_report(self, vendor, month):
        """
        Generate monthly reconciliation report for vendor billing
        """

        usage_df = pd.DataFrame(self.usage_history)

        if len(usage_df) == 0:
            return pd.DataFrame()

        # Filter by vendor and month
        vendor_usage = usage_df[
            (usage_df['vendor'] == vendor) &
            (usage_df['consumption_date'].dt.to_period('M') == month)
        ]

        # Aggregate
        summary = vendor_usage.groupby('device_id').agg({
            'serial_number': 'count',
            'cost': 'sum'
        }).rename(columns={'serial_number': 'quantity', 'cost': 'total_cost'})

        summary['average_cost'] = summary['total_cost'] / summary['quantity']

        return summary

    def aging_report(self):
        """
        Report devices sitting in consignment for extended periods
        (potential obsolescence risk)
        """

        aging_data = []

        for serial, device in self.inventory.items():
            if device['status'] == 'available':
                days_on_hand = (datetime.now() - device['received_date']).days

                aging_data.append({
                    'serial_number': serial,
                    'device_id': device['device_id'],
                    'vendor': device['vendor'],
                    'cost': device['cost'],
                    'days_on_hand': days_on_hand,
                    'risk_level': self._aging_risk(days_on_hand)
                })

        aging_df = pd.DataFrame(aging_data)

        if len(aging_df) > 0:
            aging_df = aging_df.sort_values('days_on_hand', ascending=False)

        return aging_df

    @staticmethod
    def _aging_risk(days):
        """Categorize aging risk"""
        if days > 365:
            return 'Critical'
        elif days > 180:
            return 'High'
        elif days > 90:
            return 'Medium'
        else:
            return 'Low'

# Example usage
consignment = ConsignmentInventoryManager(
    location_name="Memorial Hospital",
    par_level_policy={
        'Hip-Implant-A': 5,
        'Knee-Implant-B': 8,
        'Cardiac-Stent-C': 10
    }
)

# Add devices to consignment
for i in range(5):
    consignment.add_device(
        device_id='Hip-Implant-A',
        serial_number=f'HIP-SN-{1000+i}',
        cost=3500,
        vendor='Orthopedic Devices Inc',
        metadata={'lot': 'LOT-2024-A'}
    )

# Consume a device
usage = consignment.consume_device(
    serial_number='HIP-SN-1000',
    patient_id='PT-123456',
    procedure_info={'procedure': 'Total Hip Arthroplasty', 'surgeon': 'Dr. Smith'}
)

print(f"Device consumed: {usage['serial_number']}")
print(f"Cost: ${usage['cost']:,.2f}")

# Check PAR levels
replenishment = consignment.check_par_levels()
if replenishment:
    print("\nReplenishment needed:")
    for item in replenishment:
        print(f"  {item['device_id']}: Need {item['replenishment_qty']} units")
```

---

## Loaner Set Management

### Surgical Loaner Set Tracking

**Loaner Sets:**
- Trays of specialized instruments
- Loaned by manufacturer for specific procedures
- Must be returned after use
- High value ($50K - $500K per set)
- Tracking critical to avoid loss charges

```python
from datetime import datetime, timedelta

class LoanerSetManager:
    """
    Track loaner instrument sets for surgeries
    """

    def __init__(self):
        self.loaner_sets = {}
        self.bookings = []
        self.movements = []

    def register_loaner_set(self, set_id, vendor, description, value, contents):
        """
        Register a loaner set in the system
        """

        self.loaner_sets[set_id] = {
            'set_id': set_id,
            'vendor': vendor,
            'description': description,
            'value': value,
            'contents': contents,
            'status': 'available',
            'location': 'Vendor',
            'registered_date': datetime.now()
        }

    def book_loaner_set(self, set_id, procedure_date, surgeon, procedure_type, patient_id):
        """
        Book loaner set for upcoming surgery
        """

        if set_id not in self.loaner_sets:
            raise ValueError(f"Loaner set {set_id} not registered")

        booking = {
            'booking_id': f"BOOK-{len(self.bookings)+1}",
            'set_id': set_id,
            'procedure_date': procedure_date,
            'surgeon': surgeon,
            'procedure_type': procedure_type,
            'patient_id': patient_id,
            'booking_date': datetime.now(),
            'status': 'booked'
        }

        self.bookings.append(booking)

        # Update set status
        self.loaner_sets[set_id]['status'] = 'booked'

        return booking

    def receive_loaner_set(self, set_id, received_by, condition='good'):
        """
        Record receipt of loaner set at hospital
        """

        if set_id not in self.loaner_sets:
            raise ValueError(f"Loaner set {set_id} not registered")

        movement = {
            'set_id': set_id,
            'event_type': 'received',
            'timestamp': datetime.now(),
            'location': 'Hospital Central Sterile',
            'operator': received_by,
            'condition': condition
        }

        self.movements.append(movement)

        # Update set status and location
        self.loaner_sets[set_id]['status'] = 'in_house'
        self.loaner_sets[set_id]['location'] = 'Hospital Central Sterile'
        self.loaner_sets[set_id]['received_date'] = datetime.now()

    def send_to_sterile_processing(self, set_id, processor):
        """
        Send loaner set for sterilization
        """

        movement = {
            'set_id': set_id,
            'event_type': 'sent_to_sterilization',
            'timestamp': datetime.now(),
            'location': 'Sterile Processing',
            'operator': processor
        }

        self.movements.append(movement)

        self.loaner_sets[set_id]['location'] = 'Sterile Processing'
        self.loaner_sets[set_id]['status'] = 'processing'

    def ready_for_surgery(self, set_id, sterilization_lot):
        """
        Mark loaner set as sterile and ready
        """

        movement = {
            'set_id': set_id,
            'event_type': 'sterilization_complete',
            'timestamp': datetime.now(),
            'location': 'Sterile Storage',
            'sterilization_lot': sterilization_lot
        }

        self.movements.append(movement)

        self.loaner_sets[set_id]['location'] = 'Sterile Storage'
        self.loaner_sets[set_id]['status'] = 'ready'

    def use_in_surgery(self, set_id, procedure_info):
        """
        Record use in surgery
        """

        movement = {
            'set_id': set_id,
            'event_type': 'used_in_surgery',
            'timestamp': datetime.now(),
            'location': f"OR-{procedure_info.get('or_room', 'Unknown')}",
            'procedure_info': procedure_info
        }

        self.movements.append(movement)

        self.loaner_sets[set_id]['status'] = 'used'
        self.loaner_sets[set_id]['last_used_date'] = datetime.now()

    def return_to_vendor(self, set_id, carrier, tracking_number):
        """
        Ship loaner set back to vendor
        """

        movement = {
            'set_id': set_id,
            'event_type': 'returned_to_vendor',
            'timestamp': datetime.now(),
            'location': 'In Transit to Vendor',
            'carrier': carrier,
            'tracking_number': tracking_number
        }

        self.movements.append(movement)

        self.loaner_sets[set_id]['status'] = 'returned'
        self.loaner_sets[set_id]['location'] = 'In Transit to Vendor'
        self.loaner_sets[set_id]['return_date'] = datetime.now()

    def overdue_loaners_report(self, days_threshold=14):
        """
        Identify loaner sets held beyond expected return date
        """

        overdue = []

        for set_id, loaner in self.loaner_sets.items():
            if loaner['status'] in ['available', 'returned']:
                continue

            if 'received_date' in loaner:
                days_on_hand = (datetime.now() - loaner['received_date']).days

                if days_on_hand > days_threshold:
                    overdue.append({
                        'set_id': set_id,
                        'vendor': loaner['vendor'],
                        'description': loaner['description'],
                        'value': loaner['value'],
                        'days_on_hand': days_on_hand,
                        'status': loaner['status'],
                        'location': loaner['location']
                    })

        return pd.DataFrame(overdue)

    def loaner_utilization_report(self):
        """
        Analyze loaner set utilization and costs
        """

        utilization_data = []

        for set_id, loaner in self.loaner_sets.items():
            # Count uses
            uses = [m for m in self.movements if m['set_id'] == set_id and m['event_type'] == 'used_in_surgery']

            # Calculate average time on-site
            received_events = [m for m in self.movements if m['set_id'] == set_id and m['event_type'] == 'received']
            returned_events = [m for m in self.movements if m['set_id'] == set_id and m['event_type'] == 'returned_to_vendor']

            cycles = min(len(received_events), len(returned_events))
            if cycles > 0:
                avg_days_on_site = sum([
                    (returned_events[i]['timestamp'] - received_events[i]['timestamp']).days
                    for i in range(cycles)
                ]) / cycles
            else:
                avg_days_on_site = 0

            utilization_data.append({
                'set_id': set_id,
                'vendor': loaner['vendor'],
                'description': loaner['description'],
                'value': loaner['value'],
                'times_used': len(uses),
                'avg_days_on_site': round(avg_days_on_site, 1),
                'cycles': cycles
            })

        return pd.DataFrame(utilization_data)

# Example usage
loaner_mgr = LoanerSetManager()

# Register loaner sets
loaner_mgr.register_loaner_set(
    set_id='LOANER-SPINE-001',
    vendor='Spinal Devices Corp',
    description='Lumbar Fusion Instrument Set',
    value=125000,
    contents=['Pedicle Screws', 'Rods', 'Insertion Tools', 'Measurement Guides']
)

# Book for surgery
booking = loaner_mgr.book_loaner_set(
    set_id='LOANER-SPINE-001',
    procedure_date=datetime.now() + timedelta(days=7),
    surgeon='Dr. Williams',
    procedure_type='Lumbar Fusion L4-L5',
    patient_id='PT-789012'
)

# Track through process
loaner_mgr.receive_loaner_set('LOANER-SPINE-001', received_by='Materials Clerk')
loaner_mgr.send_to_sterile_processing('LOANER-SPINE-001', processor='SPD Tech A')
loaner_mgr.ready_for_surgery('LOANER-SPINE-001', sterilization_lot='STER-20240215-001')
loaner_mgr.use_in_surgery('LOANER-SPINE-001', {'or_room': '5', 'surgeon': 'Dr. Williams'})
loaner_mgr.return_to_vendor('LOANER-SPINE-001', carrier='FedEx', tracking_number='TRACK123456')

print("Loaner set lifecycle completed")
```

---

## Recall Management

### Medical Device Recall Process

**FDA Recall Classifications:**
- **Class I**: Reasonable probability of serious adverse health consequences or death
- **Class II**: Temporary or medically reversible adverse health consequences
- **Class III**: Not likely to cause adverse health consequences

```python
class RecallManager:
    """
    Manage medical device recalls
    """

    def __init__(self, traceability_system):
        self.traceability_system = traceability_system
        self.recalls = []

    def initiate_recall(self, recall_id, device_id, lot_numbers, reason,
                        recall_class, recall_date, corrective_action):
        """
        Initiate a device recall
        """

        recall = {
            'recall_id': recall_id,
            'device_id': device_id,
            'lot_numbers': lot_numbers,
            'reason': reason,
            'recall_class': recall_class,  # I, II, or III
            'recall_date': recall_date,
            'corrective_action': corrective_action,
            'status': 'active'
        }

        self.recalls.append(recall)

        return recall

    def identify_affected_devices(self, recall_id):
        """
        Identify all devices affected by recall using traceability
        """

        recall = next((r for r in self.recalls if r['recall_id'] == recall_id), None)

        if not recall:
            raise ValueError(f"Recall {recall_id} not found")

        affected_devices = []

        for lot_number in recall['lot_numbers']:
            # Find devices from affected lots
            serial_numbers = self.traceability_system.find_devices_by_lot(lot_number)

            for serial in serial_numbers:
                device_history = self.traceability_system.trace_device_history(serial)

                # Get current location/status
                latest_event = device_history.iloc[-1] if len(device_history) > 0 else None

                affected_devices.append({
                    'serial_number': serial,
                    'lot_number': lot_number,
                    'current_location': latest_event['location'] if latest_event is not None else 'Unknown',
                    'status': latest_event['event_type'] if latest_event is not None else 'Unknown',
                    'last_event_date': latest_event['timestamp'] if latest_event is not None else None
                })

        return pd.DataFrame(affected_devices)

    def identify_implanted_devices(self, recall_id):
        """
        Identify devices from recall that have been implanted (most critical)
        """

        affected = self.identify_affected_devices(recall_id)

        # Filter to implanted devices
        implanted = affected[affected['status'] == 'implanted']

        return implanted

    def customer_notification_list(self, recall_id):
        """
        Generate list of customers to notify about recall
        """

        affected = self.identify_affected_devices(recall_id)

        # Group by current location (customer)
        customers = affected.groupby('current_location').agg({
            'serial_number': 'count',
            'status': lambda x: ', '.join(x.unique())
        }).rename(columns={'serial_number': 'affected_devices'})

        return customers

    def recall_effectiveness_check(self, recall_id):
        """
        Track recall effectiveness (how many devices recovered)
        """

        affected = self.identify_affected_devices(recall_id)
        total_affected = len(affected)

        # Devices that have been returned/recovered
        recovered = affected[affected['status'] == 'returned']
        recovered_count = len(recovered)

        # Devices still implanted
        implanted = affected[affected['status'] == 'implanted']
        implanted_count = len(implanted)

        # Devices unaccounted for
        unaccounted = affected[~affected['status'].isin(['returned', 'implanted'])]
        unaccounted_count = len(unaccounted)

        effectiveness = {
            'recall_id': recall_id,
            'total_affected': total_affected,
            'recovered': recovered_count,
            'recovery_rate': (recovered_count / total_affected * 100) if total_affected > 0 else 0,
            'still_implanted': implanted_count,
            'unaccounted': unaccounted_count
        }

        return effectiveness

# Example usage
# Assuming traceability system from earlier example
recall_mgr = RecallManager(traceability)

# Initiate recall
recall = recall_mgr.initiate_recall(
    recall_id='RECALL-2024-001',
    device_id='10884521123456',
    lot_numbers=['LOT2024A', 'LOT2024B'],
    reason='Potential battery malfunction',
    recall_class='I',  # Class I - serious
    recall_date=datetime.now(),
    corrective_action='Return device to manufacturer for inspection and replacement'
)

print(f"Recall initiated: {recall['recall_id']}")
print(f"Class: {recall['recall_class']}")
print(f"Affected lots: {', '.join(recall['lot_numbers'])}")

# Identify affected devices
affected = recall_mgr.identify_affected_devices('RECALL-2024-001')
print(f"\nTotal affected devices: {len(affected)}")

# Identify implanted (most critical)
implanted = recall_mgr.identify_implanted_devices('RECALL-2024-001')
print(f"Implanted devices requiring patient notification: {len(implanted)}")

# Customer notification list
customers = recall_mgr.customer_notification_list('RECALL-2024-001')
print("\nCustomers to notify:")
print(customers)
```

---

## Cold Chain & Temperature Control

### Temperature-Controlled Distribution

**Requirements:**
- Certain biologics and tissue-based devices
- Maintain temperature range throughout distribution
- Continuous monitoring
- Excursion documentation

```python
import numpy as np

class ColdChainMonitor:
    """
    Monitor temperature-controlled shipments
    """

    def __init__(self, shipment_id, product_id, temp_range):
        self.shipment_id = shipment_id
        self.product_id = product_id
        self.min_temp, self.max_temp = temp_range
        self.temperature_log = []
        self.excursions = []

    def record_temperature(self, timestamp, temperature, location, recorder):
        """
        Record temperature reading
        """

        reading = {
            'timestamp': timestamp,
            'temperature': temperature,
            'location': location,
            'recorder': recorder,
            'in_range': self.min_temp <= temperature <= self.max_temp
        }

        self.temperature_log.append(reading)

        # Check for excursion
        if not reading['in_range']:
            excursion = {
                'excursion_start': timestamp,
                'temperature': temperature,
                'location': location,
                'severity': self._calculate_severity(temperature)
            }
            self.excursions.append(excursion)

        return reading

    def _calculate_severity(self, temperature):
        """
        Determine excursion severity
        """

        if temperature < self.min_temp:
            deviation = self.min_temp - temperature
        else:
            deviation = temperature - self.max_temp

        if deviation <= 2:
            return 'Minor'
        elif deviation <= 5:
            return 'Moderate'
        else:
            return 'Major'

    def compliance_report(self):
        """
        Generate temperature compliance report
        """

        if not self.temperature_log:
            return None

        df = pd.DataFrame(self.temperature_log)

        total_readings = len(df)
        compliant_readings = df['in_range'].sum()
        compliance_rate = (compliant_readings / total_readings * 100) if total_readings > 0 else 0

        report = {
            'shipment_id': self.shipment_id,
            'product_id': self.product_id,
            'required_range': f"{self.min_temp}°F - {self.max_temp}°F",
            'total_readings': total_readings,
            'compliant_readings': compliant_readings,
            'compliance_rate': round(compliance_rate, 2),
            'num_excursions': len(self.excursions),
            'min_temp_recorded': df['temperature'].min(),
            'max_temp_recorded': df['temperature'].max(),
            'avg_temp': round(df['temperature'].mean(), 2)
        }

        return report

    def excursion_summary(self):
        """
        Summarize temperature excursions
        """

        if not self.excursions:
            return pd.DataFrame()

        return pd.DataFrame(self.excursions)

# Example usage
cold_chain = ColdChainMonitor(
    shipment_id='SHIP-2024-001',
    product_id='Tissue-Graft-A',
    temp_range=(36, 46)  # 36-46°F required
)

# Simulate temperature readings
np.random.seed(42)
base_temp = 40
for hour in range(48):
    temp = base_temp + np.random.normal(0, 2)

    # Simulate excursion at hour 24
    if hour == 24:
        temp = 50  # Excursion

    cold_chain.record_temperature(
        timestamp=datetime.now() + timedelta(hours=hour),
        temperature=round(temp, 1),
        location='In Transit',
        recorder='Data Logger SN12345'
    )

# Generate reports
compliance = cold_chain.compliance_report()
print("Cold Chain Compliance:")
print(f"  Compliance Rate: {compliance['compliance_rate']}%")
print(f"  Excursions: {compliance['num_excursions']}")
print(f"  Temp Range: {compliance['min_temp_recorded']}°F - {compliance['max_temp_recorded']}°F")

if compliance['num_excursions'] > 0:
    excursions = cold_chain.excursion_summary()
    print("\nTemperature Excursions:")
    print(excursions)
```

---

## Distribution Performance Metrics

### Key Performance Indicators (KPIs)

```python
def calculate_distribution_kpis(shipment_data_df, inventory_data_df, recall_data_df=None):
    """
    Calculate medical device distribution KPIs

    Parameters:
    - shipment_data_df: Shipment transactions
    - inventory_data_df: Inventory positions
    - recall_data_df: Recall events (optional)
    """

    kpis = {}

    # On-Time Delivery Rate
    if 'on_time' in shipment_data_df.columns:
        kpis['otd_rate'] = (shipment_data_df['on_time'].sum() / len(shipment_data_df) * 100)

    # Perfect Order Rate (on-time, complete, damage-free, correct documentation)
    if all(col in shipment_data_df.columns for col in ['on_time', 'complete', 'damage_free', 'docs_correct']):
        perfect_orders = shipment_data_df[
            shipment_data_df['on_time'] &
            shipment_data_df['complete'] &
            shipment_data_df['damage_free'] &
            shipment_data_df['docs_correct']
        ]
        kpis['perfect_order_rate'] = (len(perfect_orders) / len(shipment_data_df) * 100)

    # Inventory Accuracy
    if 'physical_count' in inventory_data_df.columns and 'system_count' in inventory_data_df.columns:
        accurate_items = inventory_data_df[
            inventory_data_df['physical_count'] == inventory_data_df['system_count']
        ]
        kpis['inventory_accuracy'] = (len(accurate_items) / len(inventory_data_df) * 100)

    # UDI Compliance Rate
    if 'udi_compliant' in inventory_data_df.columns:
        kpis['udi_compliance_rate'] = (inventory_data_df['udi_compliant'].sum() / len(inventory_data_df) * 100)

    # Average Recall Response Time
    if recall_data_df is not None and not recall_data_df.empty:
        if 'response_hours' in recall_data_df.columns:
            kpis['avg_recall_response_hours'] = recall_data_df['response_hours'].mean()

    # Traceability Completeness
    if 'traceability_complete' in shipment_data_df.columns:
        kpis['traceability_completeness'] = (
            shipment_data_df['traceability_complete'].sum() / len(shipment_data_df) * 100
        )

    # Cold Chain Compliance (if applicable)
    if 'temp_compliant' in shipment_data_df.columns:
        cold_chain_shipments = shipment_data_df[shipment_data_df['requires_cold_chain'] == True]
        if len(cold_chain_shipments) > 0:
            kpis['cold_chain_compliance'] = (
                cold_chain_shipments['temp_compliant'].sum() / len(cold_chain_shipments) * 100
            )

    # Format KPIs
    for key in kpis:
        if 'rate' in key or 'compliance' in key or 'accuracy' in key or 'completeness' in key:
            kpis[key] = round(kpis[key], 2)
        else:
            kpis[key] = round(kpis[key], 2)

    return kpis

# Example usage
shipment_data = pd.DataFrame({
    'shipment_id': range(1, 101),
    'on_time': np.random.choice([True, False], 100, p=[0.95, 0.05]),
    'complete': np.random.choice([True, False], 100, p=[0.98, 0.02]),
    'damage_free': np.random.choice([True, False], 100, p=[0.99, 0.01]),
    'docs_correct': np.random.choice([True, False], 100, p=[0.97, 0.03]),
    'traceability_complete': np.random.choice([True, False], 100, p=[0.99, 0.01]),
    'requires_cold_chain': np.random.choice([True, False], 100, p=[0.15, 0.85]),
    'temp_compliant': np.random.choice([True, False], 100, p=[0.98, 0.02])
})

inventory_data = pd.DataFrame({
    'item_id': range(1, 501),
    'physical_count': np.random.randint(0, 100, 500),
    'system_count': np.random.randint(0, 100, 500),
    'udi_compliant': np.random.choice([True, False], 500, p=[0.95, 0.05])
})

# Make some matching for accuracy
inventory_data.loc[0:450, 'physical_count'] = inventory_data.loc[0:450, 'system_count']

kpis = calculate_distribution_kpis(shipment_data, inventory_data)

print("Medical Device Distribution KPIs:")
for metric, value in kpis.items():
    print(f"  {metric}: {value}{'%' if 'rate' in metric or 'compliance' in metric else ''}")
```

---

## Tools & Libraries

### Medical Device Software Systems

**ERP/QMS Systems:**
- **TrackWise**: Quality management and compliance
- **MasterControl**: Document control and quality management
- **Veeva Vault**: Regulated content management
- **SAP for Medical Devices**: ERP with device-specific functionality
- **Oracle Agile PLM**: Product lifecycle management

**Traceability & Serialization:**
- **TraceLink**: Network-based track and trace
- **Systech UniSecure**: Serialization and traceability
- **Optel Vision**: End-to-end traceability
- **Antares Vision**: Track and trace solutions
- **rfxcel**: Supply chain traceability platform

**Consignment & Loaner Management:**
- **iMDsoft**: Implant and consignment management
- **Attainia**: Healthcare supply chain management
- **SmartTray**: Loaner asset tracking
- **Surgical Information Systems (SIS)**: OR asset tracking

**Cold Chain Monitoring:**
- **Sensitech**: Temperature monitoring solutions
- **Emerson Cargo Solutions**: Cold chain visibility
- **Tive**: Real-time tracking with sensors
- **Controlant**: Temperature monitoring platform

### Python Libraries

**Serialization & Barcoding:**
- `python-barcode`: Barcode generation
- `qrcode`: QR code generation
- `pylibdmtx`: Data Matrix barcode reading
- `pyzbar`: Barcode/QR code reading

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scipy`: Statistical analysis
- `matplotlib`, `seaborn`: Visualization

**Database & Tracking:**
- `sqlalchemy`: Database ORM
- `pymongo`: MongoDB interface
- `redis-py`: Redis for caching
- `flask`/`fastapi`: API development

---

## Common Challenges & Solutions

### Challenge: UDI Compliance Implementation

**Problem:**
- Complex labeling requirements
- Multiple standards (GS1, HIBCC, ICCBBA)
- Integration with existing systems
- Historical data gaps

**Solutions:**
- Phase implementation by device class
- Implement GS1 standard (most common)
- Automated label generation systems
- Retrofit historical data where possible
- Partner with labeling vendors
- Train staff on scanning requirements
- Regular compliance audits

### Challenge: Implanted Device Traceability

**Problem:**
- Requires integration with hospital EMR
- Surgical documentation gaps
- PHI security concerns
- Real-time data capture difficult

**Solutions:**
- OR integration systems for auto-capture
- Barcode scanning at point of use
- EMR integration via HL7/FHIR
- Secure, HIPAA-compliant systems
- Simplified capture process for surgeons
- Dedicated data entry staff as backup
- Regular reconciliation audits

### Challenge: Consignment Inventory Optimization

**Problem:**
- Overstocked consignment locations
- Difficult to balance availability vs. cost
- Aging inventory risk
- Vendor relationship management

**Solutions:**
- Data-driven PAR level optimization
- Regular usage analysis by location
- Vendor scorecard for fill rates
- Transfer slow-moving stock between sites
- Demand forecasting for procedures
- Contract terms for aging inventory
- VMI (vendor-managed inventory) programs

### Challenge: Loaner Set Tracking & Returns

**Problem:**
- Lost or delayed returns
- Large financial liability
- Manual tracking processes
- Sterilization delays

**Solutions:**
- Automated loaner tracking system
- RFID tagging of loaner sets
- Alerts for overdue returns
- Dedicated loaner coordinator role
- Carrier contracts for returns
- Streamlined sterilization process
- Vendor penalties for late pickups

### Challenge: Recall Execution Speed

**Problem:**
- Difficult to locate all affected devices
- Incomplete traceability data
- Customer notification delays
- Regulatory reporting requirements

**Solutions:**
- Robust traceability system from start
- Automated recall identification
- Pre-built customer notification templates
- Recall team training and drills
- Integration with FDA recall system
- Regular traceability audits
- Mock recalls for preparedness

### Challenge: Cold Chain Compliance

**Problem:**
- Temperature excursions during transport
- Data logger failures
- Seasonal weather variations
- International shipments

**Solutions:**
- Validated packaging systems
- Redundant temperature monitoring
- Real-time alerts for excursions
- Seasonal shipping plans (avoid extreme weather)
- Qualified carriers and lanes
- Backup shipping options
- Excursion investigation protocols

---

## Output Format

### Medical Device Distribution Report

**Executive Summary:**
- Distribution network overview
- Key compliance metrics
- Performance vs. targets
- Critical issues requiring attention

**UDI & Traceability Compliance:**

| Device Category | Class | UDI Compliance | Traceability Completeness | Gap Analysis |
|----------------|-------|----------------|---------------------------|--------------|
| Cardiac Implants | III | 100% | 98% | 2% missing surgery data |
| Orthopedic Implants | III | 100% | 95% | 5% EMR integration gaps |
| Infusion Pumps | II | 98% | 100% | 2% legacy products |
| Surgical Instruments | I | 85% | N/A | 15% pending label updates |

**Consignment Inventory Summary:**

| Hospital | Device Category | On-Hand Qty | On-Hand Value | Monthly Usage | Turns | Aging >180 Days |
|----------|----------------|-------------|---------------|---------------|-------|-----------------|
| Memorial Hospital | Cardiac Stents | 45 | $67,500 | 8 | 2.1 | 3 units |
| Regional Medical | Hip Implants | 62 | $217,000 | 15 | 2.9 | 0 units |
| University Hospital | Spine Sets | 38 | $133,000 | 6 | 1.9 | 8 units |

**Active Recalls:**

| Recall ID | Device | Class | Affected Lots | Total Devices | Implanted | Recovered | Recovery Rate |
|-----------|--------|-------|---------------|---------------|-----------|-----------|---------------|
| REC-2024-001 | Pacemaker Model X | I | LOT-A, LOT-B | 247 | 89 | 158 | 64% |
| REC-2024-003 | IV Pump Series 3 | II | LOT-2024-C | 1,205 | 0 | 1,089 | 90% |

**Distribution Performance Metrics:**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| On-Time Delivery | 96.2% | 95% | ✓ |
| Perfect Order Rate | 92.8% | 95% | ⚠ |
| Inventory Accuracy | 99.1% | 98% | ✓ |
| UDI Compliance | 97.5% | 100% | ⚠ |
| Traceability Completeness | 98.3% | 99% | ⚠ |
| Cold Chain Compliance | 99.8% | 99% | ✓ |
| Recall Response Time | 6.2 hrs | < 8 hrs | ✓ |

**Action Items:**
1. Complete UDI labeling for remaining 2.5% of Class I/II devices
2. Improve EMR integration to close traceability gaps
3. Accelerate recovery efforts for Class I recall REC-2024-001
4. Reduce consignment aging inventory at University Hospital
5. Address perfect order rate gap (documentation accuracy)

---

## Questions to Ask

If you need more context:

1. What device classes are you distributing? (I, II, III)
2. What's the current state of UDI compliance?
3. Do you have a traceability system in place?
4. What percentage of business is consignment vs. purchase?
5. How many active recalls do you manage annually?
6. Are you distributing temperature-sensitive products?
7. What's your current inventory accuracy rate?
8. Do you distribute internationally?
9. What QMS system is in use?
10. What are the biggest compliance concerns or gaps?

---

## Related Skills

- **hospital-logistics**: Hospital internal supply chain management
- **pharmacy-supply-chain**: Pharmaceutical distribution and logistics
- **clinical-trial-logistics**: Clinical trial supply chain management
- **compliance-management**: Regulatory compliance and quality systems
- **track-and-trace**: Product tracking and serialization
- **quality-management**: Quality management systems
- **inventory-optimization**: Inventory optimization techniques
- **warehouse-design**: Distribution center design
- **risk-mitigation**: Supply chain risk management

---
name: pharmaceutical-supply-chain
description: When the user wants to optimize pharmaceutical supply chains, manage cold chain logistics, ensure regulatory compliance, or implement serialization. Also use when the user mentions "pharma supply chain," "GMP compliance," "cold chain," "drug serialization," "clinical trials logistics," "pharmaceutical distribution," "good distribution practices," "GDP," "drug safety," or "pharmaceutical quality." For general healthcare, see hospital-logistics. For clinical trials specifically, see clinical-trial-logistics.
---

# Pharmaceutical Supply Chain

You are an expert in pharmaceutical supply chain management, regulatory compliance, and quality systems. Your goal is to help optimize complex pharmaceutical distribution networks while ensuring patient safety, regulatory compliance, and product integrity throughout the supply chain.

## Initial Assessment

Before optimizing pharmaceutical supply chains, understand:

1. **Product Portfolio**
   - Product types? (small molecules, biologics, vaccines, medical devices)
   - Temperature requirements? (ambient, 2-8°C, frozen, ultra-cold)
   - Controlled substances? (Schedule II-V narcotics)
   - Shelf life and stability? (days, months, years)
   - Market segments? (retail pharmacy, hospital, clinical trial, specialty)

2. **Regulatory Environment**
   - Geographic markets? (US FDA, EU EMA, other regions)
   - GxP compliance level? (GMP, GDP, GCP, GLP)
   - Serialization requirements? (DSCSA, EU FMD, others)
   - Licensing requirements? (wholesale distributor, 3PL)
   - Quality system maturity? (ISO 13485, ICH guidelines)

3. **Supply Chain Structure**
   - Manufacturing sites? (API, finished goods, contract manufacturing)
   - Distribution network? (direct, wholesale, specialty distributors)
   - Cold chain capabilities?
   - Geographic footprint? (local, regional, global)
   - 3PL partnerships?

4. **Current Challenges**
   - Regulatory compliance gaps?
   - Cold chain failures or deviations?
   - Serialization implementation status?
   - Recall readiness?
   - Cost of quality issues?

---

## Pharmaceutical Supply Chain Framework

### Value Chain Structure

**Pharmaceutical Manufacturing & Distribution:**

```
API Manufacturing (Active Pharmaceutical Ingredient)
  ↓
Formulation & Fill/Finish
  ↓
Primary Packaging (bottles, vials, syringes)
  ↓
Secondary Packaging (cartons, serialization)
  ↓
Distribution Centers (ambient & cold chain)
  ↓
Wholesalers / Distributors
  ↓
Pharmacies / Hospitals / Clinics
  ↓
Patients
```

**Key Regulatory Frameworks:**

- **GMP (Good Manufacturing Practices)**: Manufacturing quality standards
- **GDP (Good Distribution Practices)**: Distribution and storage requirements
- **GCP (Good Clinical Practices)**: Clinical trial standards
- **DSCSA (Drug Supply Chain Security Act)**: US track-and-trace
- **EU FMD (Falsified Medicines Directive)**: European serialization
- **ICH (International Council for Harmonisation)**: Global standards

---

## Cold Chain Management

### Temperature Monitoring & Control

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ColdChainManager:
    """
    Manage pharmaceutical cold chain operations
    """

    def __init__(self, temperature_limits):
        """
        Initialize cold chain manager

        Parameters:
        - temperature_limits: dict with min/max temps by product type
        """
        self.temperature_limits = temperature_limits

    def validate_shipment_temperature(self, temperature_log, product_type):
        """
        Validate temperature excursions for shipment

        Parameters:
        - temperature_log: time-series temperature data
        - product_type: product classification (2-8C, frozen, etc.)

        Returns:
        - validation result with excursions
        """

        limits = self.temperature_limits.get(product_type, {})
        min_temp = limits.get('min_celsius', 2)
        max_temp = limits.get('max_celsius', 8)
        max_excursion_minutes = limits.get('max_excursion_minutes', 30)

        excursions = []
        current_excursion = None

        for idx, reading in temperature_log.iterrows():
            timestamp = reading['timestamp']
            temp = reading['temperature_celsius']

            # Check if in range
            if temp < min_temp or temp > max_temp:
                if current_excursion is None:
                    # Start new excursion
                    current_excursion = {
                        'start_time': timestamp,
                        'start_temp': temp,
                        'max_deviation': abs(temp - ((min_temp + max_temp) / 2))
                    }
                else:
                    # Continue excursion
                    deviation = abs(temp - ((min_temp + max_temp) / 2))
                    current_excursion['max_deviation'] = max(
                        current_excursion['max_deviation'], deviation
                    )
                    current_excursion['end_time'] = timestamp
                    current_excursion['end_temp'] = temp
            else:
                # Temperature back in range
                if current_excursion is not None:
                    # Complete the excursion
                    duration_minutes = (
                        current_excursion['end_time'] - current_excursion['start_time']
                    ).total_seconds() / 60

                    current_excursion['duration_minutes'] = duration_minutes
                    current_excursion['severity'] = self._classify_excursion_severity(
                        duration_minutes, current_excursion['max_deviation'],
                        max_excursion_minutes
                    )

                    excursions.append(current_excursion)
                    current_excursion = None

        # Determine overall status
        if len(excursions) == 0:
            status = 'pass'
            disposition = 'release'
        else:
            critical_excursions = [
                e for e in excursions if e['severity'] == 'critical'
            ]
            if len(critical_excursions) > 0:
                status = 'fail'
                disposition = 'reject_quarantine'
            else:
                status = 'warning'
                disposition = 'qa_review_required'

        return {
            'status': status,
            'disposition': disposition,
            'excursions': excursions,
            'excursion_count': len(excursions),
            'total_time_out_of_range_minutes': sum(
                e['duration_minutes'] for e in excursions
            )
        }

    def _classify_excursion_severity(self, duration_minutes, max_deviation,
                                     max_allowed_minutes):
        """Classify severity of temperature excursion"""

        if duration_minutes > max_allowed_minutes * 2:
            return 'critical'
        elif duration_minutes > max_allowed_minutes:
            return 'major'
        elif max_deviation > 5:  # >5°C deviation
            return 'major'
        else:
            return 'minor'

    def design_cold_chain_packaging(self, shipment_details):
        """
        Design cold chain packaging solution

        Parameters:
        - shipment_details: origin, destination, duration, product temp requirements

        Returns:
        - packaging recommendation
        """

        transit_time_hours = shipment_details['transit_time_hours']
        temp_requirement = shipment_details['temperature_requirement']
        destination_climate = shipment_details.get('destination_climate', 'temperate')

        # Determine packaging type
        if temp_requirement == 'ultra_cold':  # -80°C to -60°C
            packaging_type = 'dry_ice_shipper'
            coolant = 'dry_ice'
            qualification_duration_hours = transit_time_hours * 1.5  # 50% safety factor

        elif temp_requirement == 'frozen':  # -25°C to -10°C
            packaging_type = 'frozen_gel_pack_shipper'
            coolant = 'frozen_gel_packs'
            qualification_duration_hours = transit_time_hours * 1.3

        elif temp_requirement == '2-8C':  # Refrigerated
            if transit_time_hours <= 48:
                packaging_type = 'qualified_insulated_shipper'
                coolant = 'refrigerant_gel_packs'
            else:
                packaging_type = 'active_temp_controlled_container'
                coolant = 'active_cooling_unit'
            qualification_duration_hours = transit_time_hours * 1.2

        else:  # Ambient
            packaging_type = 'insulated_box'
            coolant = 'none'
            qualification_duration_hours = 0

        # Climate adjustment
        if destination_climate in ['tropical', 'desert'] and temp_requirement == '2-8C':
            # Need more robust solution
            packaging_type = 'active_temp_controlled_container'
            qualification_duration_hours *= 1.2

        return {
            'packaging_type': packaging_type,
            'coolant_type': coolant,
            'required_qualification_duration_hours': qualification_duration_hours,
            'temperature_monitoring': 'required' if temp_requirement != 'ambient' else 'optional',
            'data_logger_type': self._recommend_data_logger(temp_requirement),
            'estimated_cost_usd': self._estimate_packaging_cost(
                packaging_type, transit_time_hours
            )
        }

    def _recommend_data_logger(self, temp_requirement):
        """Recommend temperature data logger type"""

        if temp_requirement == 'ultra_cold':
            return 'validated_usb_logger_with_certificate'
        elif temp_requirement in ['frozen', '2-8C']:
            return 'validated_single_use_logger'
        else:
            return 'standard_logger_optional'

    def _estimate_packaging_cost(self, packaging_type, hours):
        """Estimate packaging cost"""

        costs = {
            'dry_ice_shipper': 250,
            'frozen_gel_pack_shipper': 120,
            'qualified_insulated_shipper': 80,
            'active_temp_controlled_container': 400,
            'insulated_box': 20
        }

        base_cost = costs.get(packaging_type, 50)

        # Add coolant cost based on duration
        coolant_cost = (hours / 24) * 15

        return base_cost + coolant_cost


# Example usage
temp_limits = {
    '2-8C': {'min_celsius': 2, 'max_celsius': 8, 'max_excursion_minutes': 30},
    'frozen': {'min_celsius': -25, 'max_celsius': -10, 'max_excursion_minutes': 60},
    'ultra_cold': {'min_celsius': -80, 'max_celsius': -60, 'max_excursion_minutes': 10}
}

# Simulate temperature log
temp_log = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-20 08:00', periods=100, freq='15min'),
    'temperature_celsius': np.random.normal(5, 1.5, 100)
})

# Add an excursion
temp_log.loc[30:35, 'temperature_celsius'] = [10, 11, 12, 11.5, 10, 9]

ccm = ColdChainManager(temp_limits)
validation = ccm.validate_shipment_temperature(temp_log, '2-8C')

print(f"Validation Status: {validation['status']}")
print(f"Disposition: {validation['disposition']}")
print(f"Excursions: {validation['excursion_count']}")
```

---

## Serialization & Track-and-Trace

### DSCSA Compliance Management

```python
class SerializationManager:
    """
    Manage pharmaceutical serialization and track-and-trace
    """

    def __init__(self, regulatory_region='US'):
        self.regulatory_region = regulatory_region

    def generate_serial_number(self, gtin, lot_number, sequence):
        """
        Generate serialized product identifier

        Parameters:
        - gtin: Global Trade Item Number (14 digits)
        - lot_number: Lot/batch number
        - sequence: Sequential serial number

        Returns:
        - serialized identifier
        """

        # Format: GTIN + Serial Number
        # For US DSCSA: Numeric or alphanumeric up to 20 chars

        serial = f"{sequence:010d}"  # 10-digit serial

        return {
            'gtin': gtin,
            'serial_number': serial,
            'lot_number': lot_number,
            'sscc': None,  # Serial Shipping Container Code if aggregated
            'formatted': f"(01){gtin}(21){serial}(10){lot_number}"
        }

    def create_epcis_event(self, event_type, products, location, timestamp):
        """
        Create EPCIS (Electronic Product Code Information Services) event

        Event types: commission, aggregation, observation, transformation, transaction

        Parameters:
        - event_type: type of supply chain event
        - products: list of serialized products involved
        - location: GLN (Global Location Number)
        - timestamp: event timestamp

        Returns:
        - EPCIS event structure
        """

        event = {
            'event_type': event_type,
            'event_time': timestamp.isoformat(),
            'event_timezone': 'UTC',
            'location': {
                'gln': location,
                'name': self._lookup_location_name(location)
            },
            'products': []
        }

        for product in products:
            event['products'].append({
                'gtin': product['gtin'],
                'serial_number': product['serial_number'],
                'lot_number': product['lot_number'],
                'expiry_date': product.get('expiry_date')
            })

        # Event-specific fields
        if event_type == 'commission':
            event['business_step'] = 'commissioning'
            event['disposition'] = 'active'

        elif event_type == 'shipping':
            event['business_step'] = 'shipping'
            event['disposition'] = 'in_transit'
            event['destination_gln'] = products[0].get('destination_gln')

        elif event_type == 'receiving':
            event['business_step'] = 'receiving'
            event['disposition'] = 'in_progress'

        elif event_type == 'dispensing':
            event['business_step'] = 'dispensing'
            event['disposition'] = 'dispensed'

        return event

    def _lookup_location_name(self, gln):
        """Lookup location name from GLN"""
        # Simplified - would query GLN database
        return f"Location_{gln}"

    def verify_product_authenticity(self, product_identifier, traceability_data):
        """
        Verify product authenticity using serialization data

        Parameters:
        - product_identifier: GTIN + Serial
        - traceability_data: historical EPCIS events

        Returns:
        - verification result
        """

        verification = {
            'is_authentic': True,
            'issues': [],
            'supply_chain_path': []
        }

        # Check if product was commissioned
        commission_events = [
            e for e in traceability_data
            if e['event_type'] == 'commission' and
            any(p['serial_number'] == product_identifier['serial_number']
                for p in e['products'])
        ]

        if len(commission_events) == 0:
            verification['is_authentic'] = False
            verification['issues'].append('no_commission_event_found')
            return verification

        # Trace supply chain path
        current_product = product_identifier
        path = []

        for event in sorted(traceability_data, key=lambda x: x['event_time']):
            if any(p['serial_number'] == current_product['serial_number']
                  for p in event['products']):
                path.append({
                    'event_type': event['event_type'],
                    'location': event['location']['name'],
                    'timestamp': event['event_time']
                })

        verification['supply_chain_path'] = path

        # Check for suspicious patterns
        if len(path) > 10:
            verification['issues'].append('excessive_handling_events')

        # Check for duplicates (counterfeit)
        serial_count = sum(
            1 for e in traceability_data
            if any(p['serial_number'] == current_product['serial_number']
                  for p in e['products'])
        )

        if serial_count > len(set([e['event_type'] for e in traceability_data])) * 2:
            verification['is_authentic'] = False
            verification['issues'].append('duplicate_serial_detected_possible_counterfeit')

        return verification

    def generate_recall_list(self, recall_criteria, inventory_data):
        """
        Generate list of products to recall based on criteria

        Parameters:
        - recall_criteria: lot numbers, date ranges, or serial ranges
        - inventory_data: current inventory and distribution records

        Returns:
        - list of affected products with locations
        """

        affected_products = []

        for product in inventory_data:
            match = False

            # Check lot number
            if 'lot_numbers' in recall_criteria:
                if product['lot_number'] in recall_criteria['lot_numbers']:
                    match = True

            # Check date range
            if 'manufacture_date_range' in recall_criteria:
                start, end = recall_criteria['manufacture_date_range']
                if start <= product['manufacture_date'] <= end:
                    match = True

            # Check serial range
            if 'serial_range' in recall_criteria:
                start_serial, end_serial = recall_criteria['serial_range']
                if start_serial <= product['serial_number'] <= end_serial:
                    match = True

            if match:
                affected_products.append({
                    'gtin': product['gtin'],
                    'serial_number': product['serial_number'],
                    'lot_number': product['lot_number'],
                    'current_location': product['current_location'],
                    'status': product['status'],
                    'last_movement_date': product['last_movement_date']
                })

        return pd.DataFrame(affected_products)


# Example
sm = SerializationManager(regulatory_region='US')

# Generate serial numbers
product_serial = sm.generate_serial_number(
    gtin='00312345678906',
    lot_number='LOT123456',
    sequence=1
)

print(f"Serialized Product: {product_serial['formatted']}")

# Create EPCIS event
products = [
    {'gtin': '00312345678906', 'serial_number': '0000000001',
     'lot_number': 'LOT123456', 'expiry_date': '2026-12-31'}
]

event = sm.create_epcis_event(
    event_type='commission',
    products=products,
    location='1234567890128',
    timestamp=datetime.now()
)

print(f"EPCIS Event: {event['business_step']} at {event['location']['name']}")
```

---

## Good Distribution Practices (GDP) Compliance

### Quality Management System

```python
class GDPComplianceManager:
    """
    Manage Good Distribution Practices compliance
    """

    def __init__(self):
        self.deviation_categories = [
            'temperature_excursion',
            'shipment_damage',
            'documentation_error',
            'security_breach',
            'quality_complaint'
        ]

    def log_deviation(self, deviation_details):
        """
        Log and classify quality deviation

        Parameters:
        - deviation_details: description, product, severity

        Returns:
        - deviation record with required actions
        """

        deviation_id = f"DEV_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Classify severity
        severity = self._classify_deviation_severity(deviation_details)

        # Determine required actions
        required_actions = self._determine_deviation_actions(
            deviation_details['category'],
            severity
        )

        deviation_record = {
            'deviation_id': deviation_id,
            'date_identified': datetime.now(),
            'category': deviation_details['category'],
            'description': deviation_details['description'],
            'product_affected': deviation_details.get('product_id'),
            'lot_numbers': deviation_details.get('lot_numbers', []),
            'severity': severity,
            'required_actions': required_actions,
            'status': 'open',
            'investigation_required': severity in ['critical', 'major'],
            'capa_required': severity == 'critical',  # Corrective/Preventive Action
            'regulatory_reporting_required': self._requires_regulatory_reporting(severity)
        }

        return deviation_record

    def _classify_deviation_severity(self, details):
        """Classify deviation severity"""

        category = details['category']

        # Critical: Patient safety impact
        if category == 'temperature_excursion':
            if details.get('duration_minutes', 0) > 60:
                return 'critical'
            elif details.get('duration_minutes', 0) > 30:
                return 'major'
            else:
                return 'minor'

        elif category == 'security_breach':
            return 'critical'

        elif category == 'shipment_damage':
            if details.get('product_integrity_compromised'):
                return 'critical'
            else:
                return 'major'

        else:
            return 'minor'

    def _determine_deviation_actions(self, category, severity):
        """Determine required corrective actions"""

        actions = ['document_deviation', 'notify_quality_assurance']

        if severity == 'critical':
            actions.extend([
                'quarantine_affected_products',
                'initiate_investigation_within_24hrs',
                'notify_management',
                'assess_patient_safety_impact',
                'prepare_regulatory_notification'
            ])

        elif severity == 'major':
            actions.extend([
                'quarantine_affected_products',
                'initiate_investigation_within_72hrs',
                'root_cause_analysis'
            ])

        if category == 'temperature_excursion':
            actions.append('review_temperature_monitoring_system')
            actions.append('verify_packaging_qualification')

        return actions

    def _requires_regulatory_reporting(self, severity):
        """Determine if regulatory reporting required"""
        return severity == 'critical'

    def conduct_supplier_audit(self, supplier_details):
        """
        Conduct GDP audit of pharmaceutical supplier/distributor

        Parameters:
        - supplier_details: supplier information and capabilities

        Returns:
        - audit checklist and scoring
        """

        audit_checklist = {
            'quality_system': {
                'questions': [
                    'GDP-compliant quality manual in place?',
                    'Document control system established?',
                    'Management review conducted annually?',
                    'Quality risk management process?'
                ],
                'weight': 0.20
            },
            'personnel': {
                'questions': [
                    'Qualified person designated?',
                    'GDP training program in place?',
                    'Training records maintained?',
                    'Job descriptions defined?'
                ],
                'weight': 0.15
            },
            'facilities': {
                'questions': [
                    'Temperature-controlled storage available?',
                    'Security measures adequate?',
                    'Separate quarantine area?',
                    'Clean and organized warehouse?'
                ],
                'weight': 0.15
            },
            'equipment': {
                'questions': [
                    'Temperature monitoring equipment calibrated?',
                    'Backup power systems in place?',
                    'Material handling equipment adequate?',
                    'IT systems validated?'
                ],
                'weight': 0.15
            },
            'operations': {
                'questions': [
                    'SOPs for receipt, storage, dispatch?',
                    'FIFO/FEFO system implemented?',
                    'Deviation management process?',
                    'Returns and recalls procedures?'
                ],
                'weight': 0.20
            },
            'transportation': {
                'questions': [
                    'Qualified transport providers used?',
                    'Temperature-controlled vehicles available?',
                    'Shipment validation performed?',
                    'Security measures for transport?'
                ],
                'weight': 0.15
            }
        }

        # Score each category (would be filled during actual audit)
        total_score = 0
        category_scores = {}

        for category, details in audit_checklist.items():
            # Simplified scoring - would be actual yes/no answers
            score = np.random.uniform(0.7, 1.0)  # Placeholder
            category_scores[category] = score
            total_score += score * details['weight']

        audit_result = {
            'supplier': supplier_details['name'],
            'audit_date': datetime.now(),
            'overall_score': total_score,
            'category_scores': category_scores,
            'status': 'approved' if total_score >= 0.85 else 'conditional' if total_score >= 0.70 else 'rejected',
            'critical_findings': [],
            'major_findings': [],
            'minor_findings': []
        }

        return audit_result


# Example
gdp = GDPComplianceManager()

# Log temperature deviation
deviation = gdp.log_deviation({
    'category': 'temperature_excursion',
    'description': 'Refrigerator temperature exceeded 8°C for 45 minutes',
    'product_id': 'PROD_12345',
    'lot_numbers': ['LOT_001', 'LOT_002'],
    'duration_minutes': 45
})

print(f"Deviation ID: {deviation['deviation_id']}")
print(f"Severity: {deviation['severity']}")
print(f"Required Actions: {deviation['required_actions']}")
```

---

## Clinical Trials Supply Chain

### Investigational Medicinal Product (IMP) Management

```python
class ClinicalTrialsSupplyChain:
    """
    Manage clinical trial drug supply and distribution
    """

    def __init__(self, trial_protocol):
        self.trial_protocol = trial_protocol

    def calculate_imp_demand(self, trial_sites, enrollment_plan):
        """
        Calculate Investigational Medicinal Product demand by site

        Parameters:
        - trial_sites: clinical sites with patient enrollment
        - enrollment_plan: expected enrollment over time

        Returns:
        - IMP requirements by site and time period
        """

        imp_demand = []

        for site in trial_sites:
            site_id = site['site_id']
            planned_enrollment = enrollment_plan[
                enrollment_plan['site_id'] == site_id
            ]

            for idx, period in planned_enrollment.iterrows():
                # Patients enrolled in period
                patients = period['patients']

                # Dosing regimen from protocol
                doses_per_patient = self.trial_protocol['doses_per_patient']
                treatment_duration_weeks = self.trial_protocol['treatment_duration_weeks']

                # Safety stock
                safety_stock_pct = 0.25  # 25% overage

                # Calculate requirement
                total_doses = patients * doses_per_patient
                safety_stock = total_doses * safety_stock_pct

                imp_demand.append({
                    'site_id': site_id,
                    'site_name': site['site_name'],
                    'period': period['period'],
                    'enrolled_patients': patients,
                    'required_doses': total_doses,
                    'safety_stock_doses': safety_stock,
                    'total_shipment_doses': total_doses + safety_stock
                })

        return pd.DataFrame(imp_demand)

    def generate_randomization_schedule(self, num_patients, treatment_arms,
                                       randomization_ratio):
        """
        Generate blinded randomization schedule

        Parameters:
        - num_patients: total patients to randomize
        - treatment_arms: list of treatment arms
        - randomization_ratio: ratio between arms (e.g., [1, 1] for 1:1)

        Returns:
        - randomization schedule
        """

        # Create blocks for balanced randomization
        block_size = sum(randomization_ratio)

        num_blocks = int(np.ceil(num_patients / block_size))

        randomization_schedule = []

        patient_id = 1

        for block in range(num_blocks):
            # Create one block
            block_assignments = []

            for idx, arm in enumerate(treatment_arms):
                count = randomization_ratio[idx]
                block_assignments.extend([arm] * count)

            # Randomize within block
            np.random.shuffle(block_assignments)

            # Assign to patients
            for assignment in block_assignments:
                if patient_id <= num_patients:
                    randomization_schedule.append({
                        'patient_id': f"P{patient_id:04d}",
                        'randomization_number': patient_id,
                        'treatment_arm': assignment,
                        'block_number': block + 1
                    })
                    patient_id += 1

        return pd.DataFrame(randomization_schedule)

    def optimize_depot_strategy(self, trial_sites, imp_shelf_life_months):
        """
        Optimize depot/distribution strategy for clinical trial

        Centralized vs. Regional vs. Direct-to-Site

        Parameters:
        - trial_sites: list of clinical sites with locations
        - imp_shelf_life_months: product shelf life

        Returns:
        - recommended distribution strategy
        """

        num_sites = len(trial_sites)
        geographic_spread = self._calculate_geographic_spread(trial_sites)

        # Decision logic
        if num_sites <= 5:
            strategy = 'direct_from_central'
            depots_needed = 0

        elif num_sites <= 30 and geographic_spread < 5000:  # km
            strategy = 'single_regional_depot'
            depots_needed = 1

        else:
            strategy = 'multi_regional_depots'
            depots_needed = int(num_sites / 15)  # ~15 sites per depot

        # Shelf life consideration
        if imp_shelf_life_months < 6:
            # Short shelf life requires more frequent shipments
            recommendation = f"{strategy}_with_weekly_shipments"
        else:
            recommendation = f"{strategy}_with_monthly_shipments"

        return {
            'strategy': recommendation,
            'depots_needed': depots_needed,
            'estimated_inventory_holding': self._estimate_trial_inventory(
                num_sites, strategy
            )
        }

    def _calculate_geographic_spread(self, sites):
        """Calculate geographic spread of sites (simplified)"""
        # Simplified - would use actual geocoding
        return len(sites) * 500  # Placeholder km

    def _estimate_trial_inventory(self, num_sites, strategy):
        """Estimate total inventory in supply chain"""

        if strategy == 'direct_from_central':
            pipeline_weeks = 4
        elif strategy == 'single_regional_depot':
            pipeline_weeks = 2
        else:
            pipeline_weeks = 1.5

        # Weekly demand per site (placeholder)
        weekly_demand_per_site = 50

        total_pipeline = num_sites * weekly_demand_per_site * pipeline_weeks

        return total_pipeline


# Example
trial_protocol = {
    'doses_per_patient': 52,  # Weekly dosing for 1 year
    'treatment_duration_weeks': 52
}

ct_supply = ClinicalTrialsSupplyChain(trial_protocol)

# Randomization
randomization = ct_supply.generate_randomization_schedule(
    num_patients=100,
    treatment_arms=['Drug_A', 'Placebo'],
    randomization_ratio=[1, 1]
)

print(f"Randomized {len(randomization)} patients")
print(randomization.groupby('treatment_arm').size())
```

---

## Tools & Libraries

### Python Libraries

**Supply Chain Optimization:**
- `pulp`: Optimization for distribution network
- `networkx`: Supply network modeling
- `scipy`: Statistical analysis for stability studies

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`, `seaborn`: Visualization

**Serialization:**
- `python-barcode`: Barcode generation
- `epc-tds`: EPCIS and EPC Tag Data Standard

### Commercial Software

**ERP/Supply Chain:**
- **SAP Pharma**: Pharmaceutical supply chain suite
- **Oracle Agile PLM**: Life sciences PLM
- **TraceLink**: Serialization and track-and-trace
- **Antares Vision**: End-to-end traceability

**Quality Management:**
- **Veeva Vault Quality**: Cloud QMS for life sciences
- **MasterControl**: Quality and compliance management
- **TrackWise**: CAPA and quality events
- **Sparta Systems**: Quality management

**Cold Chain:**
- **Sensitech**: Temperature monitoring and cold chain
- **Emerson Cargo Solutions**: Cold chain management
- **ELPRO**: Temperature monitoring systems
- **Tive**: Real-time tracking and monitoring

**Clinical Trials:**
- **Oracle RTSM**: Randomization and trial supply management
- **Almac RTSM**: Clinical trial supply
- **Marken**: Clinical logistics
- **World Courier**: Clinical trial shipping

---

## Common Challenges & Solutions

### Challenge: Cold Chain Failures

**Problem:**
- Temperature excursions during storage or transport
- Product integrity compromised
- Regulatory non-compliance
- Product waste and patient safety risk

**Solutions:**
- **Packaging qualification**: Test packaging for transit lanes
- **Continuous monitoring**: Real-time temperature tracking
- **Redundant systems**: Backup refrigeration and power
- **Rapid response**: Protocols for excursion investigation
- **Supplier qualification**: Audit cold chain partners
- **Temperature mapping**: Validate storage facilities
- **Seasonal testing**: Qualify for extreme weather

### Challenge: Serialization Implementation

**Problem:**
- DSCSA and EU FMD compliance requirements
- Integration with legacy systems
- High implementation costs
- Managing master data across partners

**Solutions:**
- **Phased implementation**: Start with high-value products
- **Technology selection**: Choose scalable serialization platform
- **Partner collaboration**: Align with CMOs and distributors
- **Master data management**: Centralize GTIN and product data
- **Testing**: Extensive end-to-end testing before go-live
- **Training**: Educate supply chain partners
- **Third-party services**: Consider TraceLink, rfXcel platforms

### Challenge: Drug Shortages Management

**Problem:**
- Manufacturing disruptions
- Regulatory issues halting production
- Raw material constraints
- Demand spikes (pandemic, recalls)

**Solutions:**
- **Multi-site manufacturing**: Redundant production capacity
- **Safety stock strategies**: Strategic inventory for critical drugs
- **Supply chain visibility**: Early warning systems
- **Regulatory communication**: Proactive FDA/EMA notification
- **Demand management**: Allocation to critical patients first
- **Alternative sourcing**: Qualify backup API suppliers
- **Inventory sharing**: Collaborative distribution networks

### Challenge: Controlled Substance Management

**Problem:**
- DEA Schedule II-V regulations
- Theft and diversion risk
- Complex documentation requirements
- State-by-state variations

**Solutions:**
- **Secure facilities**: Cages, vaults, access control
- **Perpetual inventory**: Real-time tracking of CS inventory
- **Dual control**: Two-person verification for transactions
- **Background checks**: Employee screening
- **Audit trails**: Complete documentation
- **Regulatory reporting**: DEA 222 forms, ARCOS reporting
- **Loss prevention**: Security measures and monitoring

### Challenge: Recall Execution Speed

**Problem:**
- Need to recall product within 24-48 hours
- Locating distributed product
- Communicating with all parties
- Ensuring complete retrieval

**Solutions:**
- **Serialization leverage**: Use track-and-trace data
- **Recall procedures**: Tested annually
- **Communication protocols**: Pre-established contact lists
- **Distribution records**: Maintained electronically
- **Mock recalls**: Practice runs quarterly
- **Batch genealogy**: Complete traceability records
- **Third-party coordination**: Align with wholesalers

---

## Output Format

### Pharmaceutical Supply Chain Report

**Executive Summary:**
- Product portfolio overview (biologics, small molecules, etc.)
- Regulatory compliance status
- Quality metrics and deviations
- Supply chain performance

**Cold Chain Performance:**

| Product | Temp Range | Shipments | Excursions | Excursion Rate | Product Loss |
|---------|------------|-----------|------------|----------------|--------------|
| Vaccine_A | 2-8°C | 1,250 | 8 | 0.64% | $12,400 |
| Biologic_B | 2-8°C | 850 | 3 | 0.35% | $45,000 |
| Insulin_C | 2-8°C | 3,200 | 15 | 0.47% | $8,200 |
| **Total** | - | **5,300** | **26** | **0.49%** | **$65,600** |

**Quality Metrics:**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| GDP Compliance | 98% | 100% | ⚠ Yellow |
| Serialization Coverage | 95% | 100% | ⚠ Yellow |
| Deviation Closure (30d) | 87% | 95% | ⚠ Yellow |
| Supplier Audit Compliance | 100% | 100% | ✓ Green |
| OTIF Delivery | 96% | 98% | ⚠ Yellow |

**Deviation Summary:**

| Severity | Count | Open | Overdue | CAPA Required |
|----------|-------|------|---------|---------------|
| Critical | 2 | 1 | 0 | 2 |
| Major | 15 | 4 | 1 | 8 |
| Minor | 42 | 12 | 3 | 0 |
| **Total** | **59** | **17** | **4** | **10** |

**Clinical Trials Status:**

| Trial ID | Phase | Sites | Patients | IMP Stock (weeks) | Issues |
|----------|-------|-------|----------|-------------------|--------|
| TRIAL_001 | III | 45 | 350 | 8 | None |
| TRIAL_002 | II | 12 | 80 | 4 | Low stock at 2 sites |
| TRIAL_003 | I | 3 | 24 | 12 | None |

**Action Items:**
1. Complete CAPA for 2 critical deviations - due by Feb 15
2. Implement backup refrigeration at DC3 - prevent future excursions
3. Complete serialization rollout for remaining 5% of products - by Q2
4. Resolve IMP stock shortage at Trial 002 sites - ship within 48 hours
5. Update GDP procedures for new EU regulations - compliance by March 1

---

## Questions to Ask

If you need more context:
1. What types of pharmaceutical products? (small molecule, biologics, vaccines)
2. What temperature requirements? (2-8°C, frozen, ultra-cold, ambient)
3. What regulatory markets? (US, EU, other regions)
4. What is the current serialization status?
5. Are there any clinical trials requiring support?
6. What are the main quality/compliance challenges?
7. What is the distribution network structure?
8. Are controlled substances involved?

---

## Related Skills

- **clinical-trial-logistics**: For investigational product management
- **cold-chain**: For temperature-controlled logistics
- **quality-management**: For QMS and GxP compliance
- **track-and-trace**: For serialization implementation
- **inventory-optimization**: For safety stock and inventory policies
- **network-design**: For distribution network optimization
- **risk-mitigation**: For supply chain risk management
- **compliance-management**: For regulatory compliance
- **medical-device-distribution**: For device-specific requirements
- **hospital-logistics**: For hospital pharmacy supply chain

---
name: pickup-delivery-problem
description: When the user wants to solve Pickup and Delivery Problems (PDP), Vehicle Routing with Pickup and Delivery (VRPPD), or handle paired pickup-delivery requests. Also use when the user mentions "PDP," "VRPPD," "pickup and delivery routing," "paired requests," "dial-a-ride," "courier routing," "moving services," or "taxi/rideshare routing." For general VRP, see vehicle-routing-problem.
---

# Pickup and Delivery Problem (PDP)

You are an expert in Pickup and Delivery Problems and paired request routing optimization. Your goal is to help design optimal routes where vehicles must pick up goods or passengers from origins and deliver them to destinations, respecting pairing constraints, precedence, and capacity throughout the route.

## Initial Assessment

Before solving PDP instances, understand:

1. **Problem Variant**
   - One-to-one (each pickup paired with delivery)?
   - Many-to-many (multiple pickups/deliveries)?
   - Dial-a-ride (passenger transportation)?
   - Same-day courier service?
   - Moving/relocation services?

2. **Pairing Constraints**
   - Hard pairing (pickup i MUST precede delivery i)?
   - Time window between pickup and delivery?
   - Maximum ride time (dial-a-ride)?
   - Can pickup/delivery be split across vehicles? (usually NO)

3. **Capacity Considerations**
   - Is capacity consumed from pickup to delivery?
   - LIFO (last-in-first-out) constraint?
   - Vehicle capacity during entire route?

4. **Temporal Constraints**
   - Time windows at pickup locations?
   - Time windows at delivery locations?
   - Maximum delivery lag after pickup?
   - Service times at each location?

5. **Problem Scale**
   - Small (< 20 requests): Exact methods possible
   - Medium (20-100 requests): Advanced heuristics
   - Large (100+ requests): Metaheuristics required

---

## Mathematical Formulation

### Pickup and Delivery VRP (PDVRP)

**Sets:**
- N = {0, 1, ..., 2n}: Nodes (0 = depot, 1..n = pickups, n+1..2n = deliveries)
- P = {1, ..., n}: Pickup nodes
- D = {n+1, ..., 2n}: Delivery nodes
- K = {1, ..., m}: Vehicles

**Parameters:**
- c_{ij}: Cost/distance from node i to j
- t_{ij}: Travel time from i to j
- s_i: Service time at node i
- q_i: Load change at node i (positive for pickup, negative for delivery)
- [e_i, l_i]: Time window at node i
- Q: Vehicle capacity

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j
- w_{ik} ≥ 0: Arrival time of vehicle k at node i
- u_{ik} ≥ 0: Load of vehicle k when leaving node i

**Objective Function:**
```
Minimize: Σ_{k∈K} Σ_{i∈N} Σ_{j∈N} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each pickup visited exactly once:
   Σ_{k∈K} Σ_{j∈N, j≠i} x_{ijk} = 1,  ∀i ∈ P

2. Each delivery visited exactly once:
   Σ_{k∈K} Σ_{j∈N, j≠i} x_{ijk} = 1,  ∀i ∈ D

3. Pickup and delivery on same vehicle:
   Σ_{j∈N, j≠i} x_{ijk} = Σ_{j∈N, j≠(n+i)} x_{(n+i)jk},  ∀i ∈ P, ∀k ∈ K

4. Pickup before delivery (precedence):
   w_{ik} + s_i + t_{i,n+i} ≤ w_{n+i,k},  ∀i ∈ P, ∀k ∈ K

5. Flow conservation:
   Σ_{i∈N, i≠h} x_{ihk} = Σ_{j∈N, j≠h} x_{hjk},  ∀h ∈ N\{0}, ∀k ∈ K

6. Time consistency:
   w_{ik} + s_i + t_{ij} ≤ w_{jk} + M*(1 - x_{ijk}),  ∀i,j ∈ N, ∀k ∈ K

7. Time windows:
   e_i ≤ w_{ik} ≤ l_i,  ∀i ∈ N, ∀k ∈ K

8. Capacity tracking:
   u_{jk} ≥ u_{ik} + q_j - Q*(1 - x_{ijk}),  ∀i,j ∈ N, ∀k ∈ K
   0 ≤ u_{ik} ≤ Q,  ∀i ∈ N, ∀k ∈ K

9. Binary variables:
   x_{ijk} ∈ {0,1}
```

---

## Exact and Heuristic Algorithms

### 1. Insertion Heuristic for PDP

```python
import numpy as np
import random

def pdp_insertion_heuristic(dist_matrix, time_matrix, requests,
                           vehicle_capacity, num_vehicles,
                           depot=0, max_route_time=480):
    """
    Sequential insertion heuristic for PDP

    Args:
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        requests: list of dicts with 'pickup_node', 'delivery_node',
                 'quantity', 'pickup_tw', 'delivery_tw'
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot: depot index
        max_route_time: maximum route duration

    Returns:
        solution dictionary
    """

    def check_feasibility(route, pickup_idx, delivery_idx,
                         pickup_pos, delivery_pos):
        """
        Check if inserting pickup and delivery is feasible

        Must check:
        - Capacity along entire route
        - Time windows
        - Precedence (pickup before delivery)
        """

        # Build temporary route
        temp_route = route.copy()
        # Insert in correct order (pickup first)
        if pickup_pos < delivery_pos:
            temp_route.insert(pickup_pos, pickup_idx)
            temp_route.insert(delivery_pos, delivery_idx)
        else:
            temp_route.insert(delivery_pos, delivery_idx)
            temp_route.insert(pickup_pos, pickup_idx)

        # Check capacity
        current_load = 0
        node_to_request = {}
        for req_idx, req in enumerate(requests):
            node_to_request[req['pickup_node']] = (req_idx, 'pickup')
            node_to_request[req['delivery_node']] = (req_idx, 'delivery')

        for node in temp_route[1:-1]:  # Skip depot
            if node in node_to_request:
                req_idx, action = node_to_request[node]
                if action == 'pickup':
                    current_load += requests[req_idx]['quantity']
                else:
                    current_load -= requests[req_idx]['quantity']

                if current_load > vehicle_capacity or current_load < 0:
                    return False

        # Check time windows and precedence
        current_time = 0
        service_times = {}  # Default service time

        for i in range(len(temp_route) - 1):
            current_node = temp_route[i]
            next_node = temp_route[i+1]

            # Travel to next node
            current_time += time_matrix[current_node][next_node]

            # Check time window
            if next_node in node_to_request:
                req_idx, action = node_to_request[next_node]
                req = requests[req_idx]

                if action == 'pickup':
                    tw = req['pickup_tw']
                else:
                    tw = req['delivery_tw']

                if current_time > tw[1]:
                    return False  # Too late

                # Wait if early
                current_time = max(current_time, tw[0])

                # Add service time
                current_time += service_times.get(next_node, 10)

        # Check total route time
        if current_time > max_route_time:
            return False

        return True

    def calculate_insertion_cost(route, pickup_idx, delivery_idx,
                                pickup_pos, delivery_pos):
        """Calculate cost increase of insertion"""

        pickup_node = requests[pickup_idx]['pickup_node']
        delivery_node = requests[delivery_idx]['delivery_node']

        # Cost of inserting pickup
        i = route[pickup_pos - 1]
        j = route[pickup_pos]
        pickup_cost = (dist_matrix[i][pickup_node] +
                      dist_matrix[pickup_node][j] -
                      dist_matrix[i][j])

        # Cost of inserting delivery (accounting for pickup already inserted)
        temp_route = route.copy()
        temp_route.insert(pickup_pos, pickup_node)

        i = temp_route[delivery_pos - 1]
        j = temp_route[delivery_pos]
        delivery_cost = (dist_matrix[i][delivery_node] +
                        dist_matrix[delivery_node][j] -
                        dist_matrix[i][j])

        return pickup_cost + delivery_cost

    # Initialize routes
    routes = [[depot, depot] for _ in range(num_vehicles)]
    unassigned_requests = list(range(len(requests)))

    # Sort requests by some criterion (e.g., earliest pickup time)
    unassigned_requests.sort(
        key=lambda r: requests[r]['pickup_tw'][0])

    # Insert requests one by one
    for req_idx in unassigned_requests[:]:
        best_route = None
        best_pickup_pos = None
        best_delivery_pos = None
        best_cost = float('inf')

        # Try inserting in each route
        for route_idx, route in enumerate(routes):
            # Try all valid insertion positions
            for pickup_pos in range(1, len(route)):
                for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                    if check_feasibility(route, req_idx, req_idx,
                                       pickup_pos, delivery_pos):
                        cost = calculate_insertion_cost(
                            route, req_idx, req_idx,
                            pickup_pos, delivery_pos)

                        if cost < best_cost:
                            best_cost = cost
                            best_route = route_idx
                            best_pickup_pos = pickup_pos
                            best_delivery_pos = delivery_pos

        # Insert request in best position
        if best_route is not None:
            pickup_node = requests[req_idx]['pickup_node']
            delivery_node = requests[req_idx]['delivery_node']

            routes[best_route].insert(best_pickup_pos, pickup_node)
            routes[best_route].insert(best_delivery_pos, delivery_node)

            unassigned_requests.remove(req_idx)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes if len(route) > 2
    )

    # Remove empty routes
    routes = [r for r in routes if len(r) > 2]

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes),
        'unassigned': unassigned_requests
    }
```

### 2. PDP with OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_pdp_ortools(locations, requests, vehicle_capacity,
                     num_vehicles, depot=0, time_limit=60):
    """
    Solve PDP using Google OR-Tools

    Args:
        locations: list of (x, y) coordinates for all locations
        requests: list of dicts:
          - pickup: pickup location index
          - delivery: delivery location index
          - quantity: load quantity
          - pickup_tw: (early, late) time window
          - delivery_tw: (early, late) time window
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot: depot index
        time_limit: time limit in seconds

    Returns:
        solution dictionary
    """
    import math

    n_locations = len(locations)

    # Build distance and time matrices
    dist_matrix = np.zeros((n_locations, n_locations))
    time_matrix = np.zeros((n_locations, n_locations))

    for i in range(n_locations):
        for j in range(n_locations):
            dist = math.sqrt((locations[i][0] - locations[j][0])**2 +
                           (locations[i][1] - locations[j][1])**2)
            dist_matrix[i][j] = dist
            time_matrix[i][j] = dist / 40 * 60  # 40 km/h in minutes

    # Create routing manager
    manager = pywrapcp.RoutingIndexManager(n_locations, num_vehicles, depot)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 100)

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    # Time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + 10)  # +10 min service

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time dimension
    routing.AddDimension(
        time_callback_index,
        30,  # allow waiting time
        3000,  # maximum time per vehicle
        False,
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')

    # Add capacity dimension with pickups and deliveries
    def demand_callback(from_index):
        """Returns the demand at the node"""
        from_node = manager.IndexToNode(from_index)
        # Check if this is a pickup or delivery
        for req in requests:
            if from_node == req['pickup']:
                return req['quantity']
            elif from_node == req['delivery']:
                return -req['quantity']
        return 0

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [vehicle_capacity] * num_vehicles,
        True,  # start cumul to zero
        'Capacity')

    # Add pickup and delivery constraints
    for request in requests:
        pickup_index = manager.NodeToIndex(request['pickup'])
        delivery_index = manager.NodeToIndex(request['delivery'])

        # Pickup and delivery must be on same route
        routing.solver().Add(
            routing.VehicleVar(pickup_index) ==
            routing.VehicleVar(delivery_index))

        # Pickup must occur before delivery
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <=
            time_dimension.CumulVar(delivery_index))

        # Add time windows
        time_dimension.CumulVar(pickup_index).SetRange(
            int(request['pickup_tw'][0]),
            int(request['pickup_tw'][1]))

        time_dimension.CumulVar(delivery_index).SetRange(
            int(request['delivery_tw'][0]),
            int(request['delivery_tw'][1]))

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_distance = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_times = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                route.append(node)
                route_times.append(solution.Value(time_var))
                index = solution.Value(routing.NextVar(index))

            route.append(manager.IndexToNode(index))
            time_var = time_dimension.CumulVar(index)
            route_times.append(solution.Value(time_var))

            if len(route) > 2:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'times': route_times
                })

                # Calculate distance
                route_distance = sum(dist_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1))
                total_distance += route_distance

        return {
            'status': 'Optimal',
            'routes': routes,
            'total_distance': total_distance,
            'num_vehicles': len(routes)
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Complete example
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # Generate PDP problem
    n_requests = 10
    depot_location = (50, 50)

    # Generate pickup and delivery locations
    pickup_locations = np.random.rand(n_requests, 2) * 100
    delivery_locations = np.random.rand(n_requests, 2) * 100

    # All locations (depot + pickups + deliveries)
    locations = [depot_location]
    locations.extend(pickup_locations.tolist())
    locations.extend(delivery_locations.tolist())

    # Create requests
    requests = []
    for i in range(n_requests):
        pickup_idx = i + 1
        delivery_idx = i + 1 + n_requests

        # Generate time windows
        pickup_early = random.randint(0, 200)
        pickup_late = pickup_early + random.randint(60, 120)
        delivery_early = pickup_late + 20
        delivery_late = delivery_early + random.randint(60, 120)

        requests.append({
            'pickup': pickup_idx,
            'delivery': delivery_idx,
            'quantity': random.randint(10, 30),
            'pickup_tw': (pickup_early, pickup_late),
            'delivery_tw': (delivery_early, delivery_late)
        })

    vehicle_capacity = 100
    num_vehicles = 4

    print(f"Problem: {n_requests} requests, {num_vehicles} vehicles")
    print(f"Capacity: {vehicle_capacity}")

    print("\nSolving PDP with OR-Tools...")
    result = solve_pdp_ortools(locations, requests, vehicle_capacity,
                              num_vehicles, time_limit=60)

    if result['status'] == 'Optimal':
        print(f"\nStatus: {result['status']}")
        print(f"Total Distance: {result['total_distance']:.2f}")
        print(f"Vehicles Used: {result['num_vehicles']}")

        print("\nRoutes:")
        for route_info in result['routes']:
            route = route_info['route']
            times = route_info['times']
            vehicle_id = route_info['vehicle_id']

            print(f"\n  Vehicle {vehicle_id + 1}:")
            print(f"    Route: {route}")

            # Identify pickups and deliveries
            for i, (node, time) in enumerate(zip(route, times)):
                if node == 0:
                    print(f"      Stop {i}: Depot at time {time:.0f}")
                elif node <= n_requests:
                    req_idx = node - 1
                    print(f"      Stop {i}: Pickup {req_idx} at time {time:.0f}")
                else:
                    req_idx = node - n_requests - 1
                    print(f"      Stop {i}: Delivery {req_idx} at time {time:.0f}")
    else:
        print(f"Status: {result['status']}")
```

---

## Tools & Libraries

- **OR-Tools (Google)**: Best for practical PDP (recommended)
- **PuLP/Pyomo**: MIP modeling
- **jsprit**: Java-based VRP solver with PDP support

---

## Common Challenges & Solutions

### Challenge: Tight Time Windows

**Problem:**
- Pickup and delivery time windows hard to satisfy
- Precedence + time windows creates difficulty

**Solutions:**
- Use time-oriented insertion criteria
- Allow some time window violations with penalties
- Increase fleet size

### Challenge: Long Ride Times (Dial-a-Ride)

**Problem:**
- Passengers have maximum ride time limits
- Hard to serve many requests efficiently

**Solutions:**
- Add ride time constraints in formulation
- Use insertion heuristics with ride time checks
- Consider direct vs. shared rides

### Challenge: LIFO Constraints

**Problem:**
- Last item picked up must be first delivered (truck loading)
- Restricts routing flexibility

**Solutions:**
- Track loading sequence explicitly
- Use specialized LIFO checking in feasibility
- May need to reject some request combinations

---

## Output Format

### PDP Solution Report

**Problem:**
- Requests: 25 pickup-delivery pairs
- Vehicles: 5 (capacity: 100 units)

**Solution:**

| Metric | Value |
|--------|-------|
| Total Distance | 892 km |
| Vehicles Used | 4 / 5 |
| Requests Served | 25 / 25 |
| On-time Pickups | 100% |
| On-time Deliveries | 100% |

**Route Details:**

**Vehicle 1:**
- Pickup 3 (8:15) → Pickup 7 (8:45) → Delivery 3 (9:20) → Delivery 7 (10:05) → Depot

---

## Questions to Ask

1. Is each pickup paired with exactly one delivery?
2. Must pickup occur before delivery on same vehicle?
3. Are there time windows at pickups/deliveries?
4. Is there max time between pickup and delivery?
5. Are there LIFO (stacking) constraints?
6. Is this dial-a-ride (passenger) or freight?

---

## Related Skills

- **vehicle-routing-problem**: For general VRP
- **vrp-time-windows**: For time window handling
- **traveling-salesman-problem**: For route sequencing


---
name: picker-routing-optimization
description: When the user wants to optimize picker routes, minimize travel distance in warehouses, or improve picking efficiency. Also use when the user mentions "pick path optimization," "warehouse routing," "travel distance minimization," "TSP in warehouses," "S-shape routing," or "optimal pick sequence." For order batching, see order-batching-optimization. For warehouse slotting, see warehouse-slotting-optimization.
---

# Picker Routing Optimization

You are an expert in warehouse picker routing and travel path optimization. Your goal is to help design optimal pick routes that minimize travel distance, reduce pick time, improve picker productivity, and maximize warehouse efficiency.

## Initial Assessment

Before optimizing picker routing, understand:

1. **Warehouse Layout**
   - Layout type (grid, diagonal, mixed)?
   - Number of aisles and length?
   - Aisle width (two-way or one-way)?
   - Cross-aisles (mid-points, ends only)?
   - Pick face configuration (single-sided, double-sided)?
   - Depot/staging location?

2. **Picking Constraints**
   - Pick method (discrete, batch, zone)?
   - Equipment (walk, picker cart, forklift, reach truck)?
   - Can skip aisles if no picks?
   - Can traverse aisles both ways?
   - Can cross aisles mid-way?
   - Pick list sequence flexibility?

3. **Order Characteristics**
   - Average picks per order/batch?
   - Pick density (picks per aisle)?
   - Pick distribution across warehouse?
   - Item location patterns?

4. **Current Performance**
   - Current routing method?
   - Average travel distance per order?
   - Picks per hour?
   - Picker feedback on routes?

---

## Picker Routing Framework

### Routing Strategies

**1. S-Shape (Traversal) Routing**
- Enter each aisle with picks, traverse completely
- Exit at far end, skip to next aisle with picks
- **Pros**: Simple, no backtracking within aisles
- **Cons**: May traverse empty portions of aisles
- **Efficiency**: Moderate (60-70% of optimal)

**2. Return Routing**
- Enter aisle, pick items, return to same end
- Move to next aisle
- **Pros**: Very simple, predictable
- **Cons**: High backtracking, longest distance
- **Efficiency**: Poor (40-50% of optimal)
- **Use**: Narrow aisles, one-way traffic only

**3. Midpoint Routing**
- If picks in front half, use return from front
- If picks in back half, traverse to back
- Requires cross-aisle in middle
- **Pros**: Better than pure S-shape or return
- **Cons**: Requires cross-aisle infrastructure
- **Efficiency**: Good (70-80% of optimal)

**4. Largest Gap Routing**
- Identify largest gap between picks in aisle
- Enter/exit to avoid traversing largest gap
- **Pros**: Adapts to pick distribution
- **Cons**: More complex, requires calculation
- **Efficiency**: Very good (80-90% of optimal)

**5. Optimal Routing (TSP-based)**
- Solve as Traveling Salesman Problem
- Find shortest path visiting all picks
- **Pros**: Best possible route
- **Cons**: Complex computation (NP-hard)
- **Efficiency**: Optimal (100%)

### Routing Objectives

```
Primary Goal:
  Minimize total travel distance

Secondary Goals:
  - Minimize pick time (travel + access)
  - Balance picker workload
  - Respect aisle traffic constraints
  - Maintain pick accuracy (logical sequence)

Constraints:
  - Aisle layout (can't cut through racks)
  - One-way aisles (directional constraints)
  - Congestion (avoid other pickers)
  - Equipment limitations (turning radius, height)
```

---

## Mathematical Formulation

### Warehouse as a Graph

Model warehouse as a directed graph G = (V, E):

**Vertices (V):**
- Pick locations
- Aisle endpoints
- Cross-aisle intersections
- Depot (start/end point)

**Edges (E):**
- Travel segments between vertices
- Edge weights = distance or time
- Directed edges for one-way aisles

**Routing Problem:**
```
Find shortest path from depot visiting all pick locations
and returning to depot

This is a variant of the Traveling Salesman Problem (TSP)
with special structure (rectilinear geometry)
```

### TSP Formulation for Warehouse

**Decision Variables:**
- x[i,j] = 1 if picker travels from location i to j, 0 otherwise
- u[i] = position of location i in route (for subtour elimination)

**Parameters:**
- d[i,j] = distance from location i to j
- P = set of pick locations
- depot = start/end point

**Objective:**

```
Minimize: Σ Σ (d[i,j] × x[i,j])  for all i,j in (P ∪ {depot})
```

**Constraints:**

```python
# 1. Leave each location exactly once (except depot twice - start and end)
for i in P:
    Σ x[i,j] = 1  for all j != i

# 2. Enter each location exactly once
for j in P:
    Σ x[i,j] = 1  for all i != j

# 3. Flow conservation
for k in P:
    Σ x[i,k] = Σ x[k,j]  for all i,j

# 4. Start and end at depot
Σ x[depot,j] = 1  for all j in P
Σ x[i,depot] = 1  for all i in P

# 5. Subtour elimination (Miller-Tucker-Zemlin)
for i,j in P:
    u[i] - u[j] + n × x[i,j] <= n - 1

# 6. Aisle constraints (can't pass through racks)
for i,j not adjacent:
    if no path exists:
        x[i,j] = 0
```

---

## Routing Algorithms

### S-Shape Routing

```python
import numpy as np
import pandas as pd

def s_shape_routing(picks, aisles, cross_aisle_locations):
    """
    S-Shape routing algorithm

    Parameters:
    -----------
    picks : DataFrame
        Columns: pick_id, aisle, position_in_aisle, side (left/right)
    aisles : dict
        {aisle_id: {'length': length, 'width': width}}
    cross_aisle_locations : dict
        {aisle_id: [positions...]}  # Where cross-aisles exist

    Returns:
    --------
    Route sequence and total distance
    """

    # Group picks by aisle
    picks_by_aisle = picks.groupby('aisle')

    route = []
    total_distance = 0
    current_position = 0  # Start at position 0 (front of warehouse)
    current_aisle = 0  # Start at aisle 0

    # Get aisles with picks (sorted)
    aisles_with_picks = sorted(picks_by_aisle.groups.keys())

    for aisle_id in aisles_with_picks:
        aisle_picks = picks_by_aisle.get_group(aisle_id).sort_values('position_in_aisle')

        # Move to aisle (cross-aisle travel)
        cross_aisle_distance = abs(aisle_id - current_aisle) * aisles[0].get('width', 10)
        total_distance += cross_aisle_distance

        # Determine entry point (front or back)
        # For S-shape: alternate front and back entry
        aisle_index = aisles_with_picks.index(aisle_id)

        if aisle_index % 2 == 0:
            # Enter from front, traverse to back
            entry_position = 0
            exit_position = aisles[aisle_id]['length']
            picks_order = aisle_picks.sort_values('position_in_aisle')
        else:
            # Enter from back, traverse to front
            entry_position = aisles[aisle_id]['length']
            exit_position = 0
            picks_order = aisle_picks.sort_values('position_in_aisle', ascending=False)

        # Move to entry point
        total_distance += abs(entry_position - current_position)

        # Traverse aisle and pick items
        for idx, pick in picks_order.iterrows():
            route.append({
                'pick_id': pick['pick_id'],
                'aisle': aisle_id,
                'position': pick['position_in_aisle'],
                'sequence': len(route) + 1
            })

        # Add traversal distance
        total_distance += abs(exit_position - entry_position)

        current_position = exit_position
        current_aisle = aisle_id

    # Return to depot
    total_distance += abs(current_position - 0)
    total_distance += abs(current_aisle - 0) * aisles[0].get('width', 10)

    return {
        'route': pd.DataFrame(route),
        'total_distance': total_distance,
        'num_picks': len(picks)
    }


# Example usage
picks = pd.DataFrame({
    'pick_id': [f'P{i:03d}' for i in range(1, 21)],
    'aisle': np.random.randint(1, 11, 20),  # 10 aisles
    'position_in_aisle': np.random.uniform(0, 100, 20),  # 100 ft aisles
    'side': np.random.choice(['left', 'right'], 20)
})

aisles = {i: {'length': 100, 'width': 12} for i in range(1, 11)}
cross_aisles = {i: [0, 100] for i in range(1, 11)}  # Front and back only

result = s_shape_routing(picks, aisles, cross_aisles)
print(f"S-Shape Routing:")
print(f"Total Distance: {result['total_distance']:.2f} ft")
print(f"Number of Picks: {result['num_picks']}")
print(f"\nFirst 5 picks in sequence:")
print(result['route'].head())
```

### Largest Gap Routing

```python
def largest_gap_routing(picks, aisles):
    """
    Largest gap routing algorithm

    For each aisle:
    - If picks only in front half: enter and return from front
    - If picks only in back half: enter and return from back
    - If picks in both halves: enter from one side, exit from other (S-shape)
      but choose entry/exit to minimize distance

    Parameters:
    -----------
    picks : DataFrame
        Pick locations
    aisles : dict
        Aisle specifications

    Returns:
    --------
    Optimized route
    """

    picks_by_aisle = picks.groupby('aisle')

    route = []
    total_distance = 0
    current_position = 0
    current_aisle = 0

    aisles_with_picks = sorted(picks_by_aisle.groups.keys())

    for aisle_id in aisles_with_picks:
        aisle_picks = picks_by_aisle.get_group(aisle_id).sort_values('position_in_aisle')
        aisle_length = aisles[aisle_id]['length']

        # Find largest gap between consecutive picks
        positions = sorted(aisle_picks['position_in_aisle'].tolist())

        # Add aisle endpoints to gap calculation
        gaps = []
        gaps.append({'gap_size': positions[0] - 0,
                    'start': 0, 'end': positions[0]})

        for i in range(len(positions) - 1):
            gaps.append({'gap_size': positions[i+1] - positions[i],
                        'start': positions[i], 'end': positions[i+1]})

        gaps.append({'gap_size': aisle_length - positions[-1],
                    'start': positions[-1], 'end': aisle_length})

        largest_gap = max(gaps, key=lambda x: x['gap_size'])

        # Determine routing strategy based on largest gap location
        if largest_gap['start'] == 0:
            # Largest gap at front, enter from back
            entry_position = aisle_length
            exit_position = aisle_length
            picks_order = aisle_picks.sort_values('position_in_aisle', ascending=False)

        elif largest_gap['end'] == aisle_length:
            # Largest gap at back, enter from front
            entry_position = 0
            exit_position = 0
            picks_order = aisle_picks.sort_values('position_in_aisle')

        else:
            # Largest gap in middle, traverse around it
            # Enter from front, exit from back (or vice versa)
            # Choose direction that minimizes total travel

            # Option 1: Front to back
            dist1 = aisle_length - largest_gap['gap_size']

            # Option 2: Back to front
            dist2 = aisle_length - largest_gap['gap_size']

            # Both same for middle gaps, so use S-shape logic
            if current_position < aisle_length / 2:
                entry_position = 0
                exit_position = aisle_length
                picks_order = aisle_picks.sort_values('position_in_aisle')
            else:
                entry_position = aisle_length
                exit_position = 0
                picks_order = aisle_picks.sort_values('position_in_aisle', ascending=False)

        # Move to aisle
        cross_aisle_distance = abs(aisle_id - current_aisle) * aisles[0].get('width', 10)
        total_distance += cross_aisle_distance

        # Move to entry point
        total_distance += abs(entry_position - current_position)

        # Pick items
        for idx, pick in picks_order.iterrows():
            route.append({
                'pick_id': pick['pick_id'],
                'aisle': aisle_id,
                'position': pick['position_in_aisle'],
                'sequence': len(route) + 1
            })

        # Calculate actual travel distance in aisle (excluding largest gap if avoided)
        if entry_position == exit_position:
            # Return routing
            furthest_pick = picks_order.iloc[-1]['position_in_aisle']
            aisle_travel = 2 * abs(furthest_pick - entry_position)
        else:
            # Traversal routing (minus largest gap if skipped)
            aisle_travel = abs(exit_position - entry_position)

        total_distance += aisle_travel

        current_position = exit_position
        current_aisle = aisle_id

    # Return to depot
    total_distance += abs(current_position - 0)
    total_distance += abs(current_aisle - 0) * aisles[0].get('width', 10)

    return {
        'route': pd.DataFrame(route) if route else pd.DataFrame(),
        'total_distance': total_distance,
        'num_picks': len(picks)
    }


result_lg = largest_gap_routing(picks, aisles)
print(f"\nLargest Gap Routing:")
print(f"Total Distance: {result_lg['total_distance']:.2f} ft")
print(f"Improvement vs S-Shape: {(result['total_distance'] - result_lg['total_distance']) / result['total_distance'] * 100:.1f}%")
```

### TSP-Based Optimal Routing

```python
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def tsp_optimal_routing(picks, warehouse_graph):
    """
    Optimal routing using TSP solver (Google OR-Tools)

    Parameters:
    -----------
    picks : DataFrame
        Pick locations with x, y coordinates
    warehouse_graph : dict
        Distance matrix considering warehouse layout constraints

    Returns:
    --------
    Optimal route
    """

    # Create distance matrix
    # In real warehouse, distance is constrained by aisles (not Euclidean)
    # Use warehouse_graph or calculate Manhattan distance

    locations = picks[['x', 'y']].values
    n_locations = len(locations)

    # Add depot (0, 0)
    depot_location = np.array([[0, 0]])
    all_locations = np.vstack([depot_location, locations])

    # Calculate distance matrix (Manhattan distance for warehouse)
    def manhattan_distance(loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    n = len(all_locations)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = manhattan_distance(
                    all_locations[i], all_locations[j]
                )

    # Convert to integer (OR-Tools requirement)
    distance_matrix = (distance_matrix * 100).astype(int)

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n locations, 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 10

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # Extract route
        route = []
        index = routing.Start(0)
        total_distance = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:  # Skip depot at start
                route.append({
                    'pick_id': picks.iloc[node - 1]['pick_id'],
                    'x': picks.iloc[node - 1]['x'],
                    'y': picks.iloc[node - 1]['y'],
                    'sequence': len(route) + 1
                })

            next_index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(index, next_index, 0)
            index = next_index

        return {
            'route': pd.DataFrame(route),
            'total_distance': total_distance / 100,  # Convert back from integer
            'num_picks': len(picks),
            'optimal': True
        }

    return {'route': pd.DataFrame(), 'total_distance': 0, 'optimal': False}


# Example with coordinates
picks_with_coords = pd.DataFrame({
    'pick_id': [f'P{i:03d}' for i in range(1, 21)],
    'x': np.random.uniform(0, 100, 20),  # x coordinate (across aisles)
    'y': np.random.uniform(0, 100, 20),  # y coordinate (down aisle)
})

# Note: In real implementation, would use actual warehouse graph
# that respects aisle structure

result_tsp = tsp_optimal_routing(picks_with_coords, warehouse_graph={})
if result_tsp['optimal']:
    print(f"\nTSP Optimal Routing:")
    print(f"Total Distance: {result_tsp['total_distance']:.2f} ft")
    print(f"First 5 picks:")
    print(result_tsp['route'].head())
```

---

## Advanced Routing Techniques

### Dynamic Routing with Congestion Avoidance

```python
class DynamicRouter:
    """
    Dynamic routing that adapts to warehouse congestion
    """

    def __init__(self, warehouse_layout, real_time_tracking=False):
        self.warehouse_layout = warehouse_layout
        self.real_time_tracking = real_time_tracking
        self.picker_locations = {}  # {picker_id: (aisle, position)}
        self.congestion_map = {}  # {aisle: congestion_level}

    def update_picker_location(self, picker_id, aisle, position):
        """Update picker location for congestion tracking"""
        self.picker_locations[picker_id] = (aisle, position)
        self.update_congestion_map()

    def update_congestion_map(self):
        """Calculate congestion level for each aisle"""
        aisle_counts = {}
        for picker_id, (aisle, position) in self.picker_locations.items():
            aisle_counts[aisle] = aisle_counts.get(aisle, 0) + 1

        # Congestion level: number of pickers / aisle capacity
        for aisle in self.warehouse_layout['aisles']:
            count = aisle_counts.get(aisle, 0)
            capacity = 3  # Assume max 3 pickers per aisle comfortably
            self.congestion_map[aisle] = count / capacity

    def route_with_congestion(self, picks, picker_id):
        """
        Calculate route considering current congestion

        Penalize aisles with high congestion
        """

        # Start with base routing algorithm (e.g., largest gap)
        base_route = largest_gap_routing(picks, self.warehouse_layout['aisles'])

        if not self.real_time_tracking:
            return base_route

        # Adjust route based on congestion
        # Re-sequence to visit less congested aisles first

        route_df = base_route['route'].copy()
        route_df['aisle'] = picks.set_index('pick_id').loc[route_df['pick_id'], 'aisle'].values
        route_df['congestion'] = route_df['aisle'].map(self.congestion_map).fillna(0)

        # Sort picks within each priority group by congestion
        route_df = route_df.sort_values(['congestion', 'aisle'])
        route_df['sequence'] = range(1, len(route_df) + 1)

        return {
            'route': route_df,
            'total_distance': base_route['total_distance'] * 1.05,  # Slight penalty for re-sequencing
            'congestion_adjusted': True
        }


# Example usage
warehouse_layout = {
    'aisles': list(range(1, 11)),
    'aisle_specs': {i: {'length': 100, 'width': 12} for i in range(1, 11)}
}

router = DynamicRouter(warehouse_layout, real_time_tracking=True)

# Simulate other pickers
router.update_picker_location('picker_1', aisle=3, position=50)
router.update_picker_location('picker_2', aisle=3, position=30)
router.update_picker_location('picker_3', aisle=7, position=60)

# Route new picker
route_dynamic = router.route_with_congestion(picks, 'picker_4')
print("\nDynamic Routing with Congestion:")
print(f"Congestion Map: {router.congestion_map}")
```

### Multi-Level Warehouse Routing

```python
def multi_level_routing(picks, levels, vertical_travel_time=30):
    """
    Routing for multi-level warehouses (mezzanines, multi-story)

    Parameters:
    -----------
    picks : DataFrame
        Picks with level, aisle, position
    levels : list
        Available levels
    vertical_travel_time : float
        Seconds to move between levels

    Returns:
    --------
    Optimized route considering vertical travel
    """

    # Group picks by level
    picks_by_level = picks.groupby('level')

    # Strategy: Minimize level changes
    # Visit all picks on one level before moving to next

    route = []
    total_distance = 0
    total_time = 0
    current_level = 1  # Start at ground level

    # Determine optimal level sequence
    # Heuristic: Ground level first, then upper levels in order

    level_sequence = sorted(picks_by_level.groups.keys())

    for level in level_sequence:
        level_picks = picks_by_level.get_group(level)

        # Move to level (vertical travel)
        if level != current_level:
            level_changes = abs(level - current_level)
            total_time += level_changes * vertical_travel_time
            current_level = level

        # Route within level (use largest gap or TSP)
        level_route = largest_gap_routing(
            level_picks,
            aisles={i: {'length': 100, 'width': 12} for i in range(1, 11)}
        )

        # Add to total route
        level_route_df = level_route['route']
        level_route_df['level'] = level
        route.append(level_route_df)

        total_distance += level_route['total_distance']

    # Combine all levels
    full_route = pd.concat(route, ignore_index=True)
    full_route['sequence'] = range(1, len(full_route) + 1)

    return {
        'route': full_route,
        'total_distance': total_distance,
        'total_time': total_time,
        'num_picks': len(picks),
        'levels_visited': len(level_sequence)
    }


# Example
picks_multi_level = pd.DataFrame({
    'pick_id': [f'P{i:03d}' for i in range(1, 31)],
    'level': np.random.choice([1, 2, 3], 30),
    'aisle': np.random.randint(1, 11, 30),
    'position_in_aisle': np.random.uniform(0, 100, 30),
})

result_ml = multi_level_routing(picks_multi_level, levels=[1, 2, 3])
print("\nMulti-Level Routing:")
print(f"Total Distance: {result_ml['total_distance']:.2f} ft")
print(f"Vertical Travel Time: {result_ml['total_time']:.0f} sec")
print(f"Levels Visited: {result_ml['levels_visited']}")
```

---

## Tools & Libraries

### Routing Software

**Warehouse Management Systems:**
- **Manhattan WMS**: Optimized pick path generation
- **Blue Yonder (JDA) WMS**: AI-driven routing
- **SAP EWM**: Pick-by-order and pick-by-wave routing
- **HighJump WMS**: Dynamic routing with real-time updates

**Specialized Optimization:**
- **Lucas Systems**: Voice-directed picking with optimized routes
- **Voxware**: Voice picking with intelligent routing
- **Google OR-Tools**: Open-source routing optimization
- **Gurobi/CPLEX**: Commercial optimization solvers

### Python Libraries

```python
# Routing Optimization
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from scipy.optimize import linear_sum_assignment
import networkx as nx  # Graph algorithms

# TSP Solvers
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

# Distance Calculations
from scipy.spatial.distance import pdist, squareform, cityblock
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```

---

## Common Challenges & Solutions

### Challenge: One-Way Aisles

**Problem:**
- Narrow aisles, only one-way traffic
- Can't use S-shape (can't traverse both directions)
- Must return from entry point

**Solutions:**
- Use return routing as baseline
- Optimize entry/exit points (front vs back)
- Use midpoint if cross-aisle available
- Consider widening key aisles for two-way traffic
- Implement aisle direction alternation (odd/even)

### Challenge: Congestion and Blocking

**Problem:**
- Multiple pickers in same aisle
- Blocking and waiting time
- Routes become longer due to avoidance

**Solutions:**
- Real-time routing with congestion awareness
- Stagger picker start times (offset by 5-10 min)
- Zone-based picking (dedicate aisles to pickers)
- Dynamic re-routing via mobile app/voice system
- Traffic flow analysis to identify bottlenecks

### Challenge: Large Pick Lists

**Problem:**
- 100+ pick locations in single order/batch
- TSP becomes computationally expensive
- Real-time routing not feasible

**Solutions:**
- Use heuristics (largest gap, S-shape) instead of TSP
- Divide into smaller batches
- Pre-compute routes offline (for recurring orders)
- Use approximate TSP (LKH, Christofides)
- Time limit on optimization (best route in 5 seconds)

### Challenge: Variable Pick Times

**Problem:**
- Some picks take 5 seconds, others 60 seconds
- Heavy items, high shelves, quantity picks
- Distance-based routing ignores time variability

**Solutions:**
- Weight edges by time, not distance
- Include pick time in route optimization
- Sequence difficult picks first (when picker fresh)
- Pre-stage heavy/bulky items (separate workflow)
- Use time-motion studies to calibrate

### Challenge: Layout Changes

**Problem:**
- Warehouse layout changes (slotting refresh)
- Routes become suboptimal
- Pickers confused by location changes

**Solutions:**
- Re-compute routes after slotting changes
- Gradual rollout (zone by zone)
- Update WMS location master immediately
- Train pickers on new layout
- Use RF/voice to direct (location-agnostic)

---

## Output Format

### Picker Routing Report

**Route Optimization Analysis - Order #12345**

**Pick List Summary:**
- Total Picks: 42 lines
- Aisles Involved: 8 aisles (2, 4, 5, 7, 9, 11, 13, 15)
- Warehouse Zones: A (18 picks), B (15 picks), C (9 picks)

**Routing Comparison:**

| Method | Distance (ft) | Est. Time (min) | Improvement |
|--------|---------------|-----------------|-------------|
| Return Routing | 1,845 | 28.5 | Baseline |
| S-Shape Routing | 1,124 | 17.4 | 39% |
| Largest Gap | 892 | 13.8 | 52% |
| TSP Optimal | 834 | 12.9 | 55% |

**Recommended Route (Largest Gap):**

```
Sequence | Pick ID | SKU | Aisle | Position | Side | Qty |
---------|---------|-----|-------|----------|------|-----|
1 | P001 | SKU_A | 2 | 15.3 | L | 2 |
2 | P002 | SKU_B | 2 | 28.7 | R | 1 |
3 | P003 | SKU_C | 2 | 45.2 | L | 3 |
4 | [Cross to Aisle 4]
5 | P008 | SKU_H | 4 | 82.1 | R | 1 |
6 | P009 | SKU_I | 4 | 67.3 | L | 2 |
...
```

**Route Visualization:**

```
Depot (Start)
  ↓
Aisle 2 [Enter Front → Pick 3 items → Exit Front]
  ↓
Cross-Aisle Travel (2 → 4)
  ↓
Aisle 4 [Enter Back → Pick 5 items → Exit Front]
  ↓
Cross-Aisle Travel (4 → 5)
  ↓
Aisle 5 [Enter Front → Pick 4 items → Exit Front]
...
  ↓
Return to Depot (End)

Total Distance: 892 ft
Estimated Time: 13.8 minutes (assuming 100 ft/min walk + 10 sec/pick)
```

**Performance Metrics:**
- Distance per Pick: 21.2 ft/pick
- Estimated Picks per Hour: 182 (vs. 145 with return routing)
- Productivity Improvement: +26%

---

## Questions to Ask

If you need more context:
1. What's your warehouse layout (grid, cross-aisles)?
2. Are aisles one-way or two-way?
3. What picking method (discrete, batch, zone)?
4. What's your average picks per order/batch?
5. What routing method do you currently use?
6. What's your current picks per hour?
7. Do you have WMS with routing capability?
8. Any congestion or blocking issues?

---

## Related Skills

- **traveling-salesman-problem**: For TSP algorithms and theory
- **order-batching-optimization**: For creating optimal batches to route
- **warehouse-slotting-optimization**: For SKU placement affecting routes
- **wave-planning-optimization**: For wave design impacting routing
- **network-flow-optimization**: For warehouse flow modeling
- **graph-algorithms**: For shortest path and routing
- **metaheuristic-optimization**: For large-scale routing problems

---
name: planogram-optimization
description: When the user wants to optimize store planograms, shelf space allocation, or visual merchandising layout. Also use when the user mentions "planogram," "shelf space optimization," "space productivity," "category management," "shelf allocation," "fixture planning," "facings optimization," or "merchandising layout." For inventory allocation, see retail-allocation. For assortment planning, see seasonal-planning.
---

# Planogram Optimization

You are an expert in retail planogram optimization and space management. Your goal is to help retailers maximize sales and profitability per square foot by optimally allocating shelf space, determining product facings, and designing efficient store layouts that balance product visibility, customer experience, and operational efficiency.

## Initial Assessment

Before optimizing planograms, understand:

1. **Store Context**
   - What store format? (grocery, apparel, electronics, pharmacy)
   - Store size and layout? (square footage, number of fixtures)
   - Traffic patterns? (entrance location, checkout placement)
   - Target customer demographics?
   - Store location type? (urban, suburban, mall)

2. **Category Characteristics**
   - What category/department needs optimization?
   - Number of SKUs in category?
   - Product dimensions? (height, width, depth)
   - Unit movement rates? (fast vs. slow movers)
   - Margin by SKU?
   - Shelf life considerations? (perishable, seasonal)

3. **Current Performance**
   - Current sales per square foot?
   - Out-of-stock frequency?
   - Space productivity by fixture?
   - Customer satisfaction with layout?
   - Labor cost for restocking?

4. **Business Objectives**
   - Maximize revenue or profit?
   - Target service level? (stock availability)
   - Cross-merchandising goals?
   - Brand/promotional requirements?
   - Operational constraints? (restocking frequency, labor)

---

## Planogram Optimization Framework

### Space Productivity Principles

**1. Space Elasticity**
- Relationship between shelf space and sales
- Diminishing returns: more space doesn't always = more sales
- Optimal facings per SKU varies by product

**2. Space Allocation Rules**
- **High-turnover items**: More facings, eye-level placement
- **High-margin items**: Premium placement
- **Impulse items**: End caps, checkout
- **Destination items**: Can be placed in back (draws traffic)
- **Complementary items**: Cross-merchandising clusters

**3. Shelf Height Effects**
- **Eye level (4-5 ft)**: Prime real estate, 40% of sales
- **Chest level (3-4 ft)**: Secondary prime, 30% of sales
- **Waist level (2-3 ft)**: Third tier, 20% of sales
- **Floor level (0-2 ft)**: Low visibility, 10% of sales
- **Above eye (5-6 ft)**: Overflow, occasional purchases

**4. Product Adjacency**
- Related products together (pasta + sauce)
- Color blocking for visual appeal
- Size progression (small to large)
- Price progression (low to high)

---

## Space-to-Sales Analysis

### Space Elasticity Modeling

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SpaceElasticityAnalyzer:
    """
    Analyze space elasticity - relationship between shelf space and sales

    Space Elasticity = % change in sales / % change in shelf space
    """

    def __init__(self, historical_data):
        """
        Parameters:
        - historical_data: DataFrame with space/sales experiments
          columns: ['sku', 'period', 'facings', 'sales_units', 'sales_dollars']
        """
        self.data = historical_data

    def calculate_space_elasticity(self, sku):
        """
        Calculate space elasticity coefficient

        Using log-log regression: log(Sales) = a + b * log(Facings)
        b is the space elasticity
        """

        sku_data = self.data[self.data['sku'] == sku].copy()

        if len(sku_data) < 5:
            return {'error': 'Insufficient data'}

        # Log transformation
        sku_data['log_facings'] = np.log(sku_data['facings'])
        sku_data['log_sales'] = np.log(sku_data['sales_units'] + 1)

        # Linear regression in log-log space
        X = sku_data['log_facings'].values.reshape(-1, 1)
        y = sku_data['log_sales'].values

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        elasticity = model.coef_[0]
        r_squared = model.score(X, y)

        # Interpretation
        if elasticity > 0.8:
            interpretation = 'High elasticity - sales very responsive to space'
        elif elasticity > 0.4:
            interpretation = 'Moderate elasticity - typical'
        elif elasticity > 0.1:
            interpretation = 'Low elasticity - limited response to space'
        else:
            interpretation = 'Very low elasticity - sales not space-dependent'

        return {
            'sku': sku,
            'elasticity': elasticity,
            'r_squared': r_squared,
            'interpretation': interpretation,
            'model': model
        }

    def estimate_sales_at_facings(self, sku, target_facings):
        """
        Estimate sales at a given number of facings

        Uses fitted elasticity model
        """

        elasticity_result = self.calculate_space_elasticity(sku)

        if 'error' in elasticity_result:
            return None

        model = elasticity_result['model']
        log_facings = np.log(target_facings)

        log_sales_pred = model.predict([[log_facings]])[0]
        estimated_sales = np.exp(log_sales_pred) - 1

        return max(0, estimated_sales)

    def find_optimal_facings(self, sku, cost_per_facing, profit_per_unit,
                            max_facings=20):
        """
        Find optimal number of facings to maximize profit

        Balance: More facings = more sales but higher space cost
        """

        elasticity_result = self.calculate_space_elasticity(sku)

        if 'error' in elasticity_result:
            return {'error': 'Cannot optimize without elasticity data'}

        facings_range = range(1, max_facings + 1)
        results = []

        for facings in facings_range:
            estimated_sales = self.estimate_sales_at_facings(sku, facings)

            # Calculate profit
            revenue = estimated_sales * profit_per_unit
            space_cost = facings * cost_per_facing

            profit = revenue - space_cost

            results.append({
                'facings': facings,
                'estimated_sales': estimated_sales,
                'revenue': revenue,
                'space_cost': space_cost,
                'profit': profit,
                'profit_per_facing': profit / facings if facings > 0 else 0
            })

        results_df = pd.DataFrame(results)

        # Find optimal
        optimal_idx = results_df['profit'].idxmax()
        optimal = results_df.iloc[optimal_idx]

        return {
            'optimal_facings': optimal['facings'],
            'expected_sales': optimal['estimated_sales'],
            'expected_profit': optimal['profit'],
            'elasticity': elasticity_result['elasticity'],
            'all_scenarios': results_df
        }

# Example usage
np.random.seed(42)

# Generate sample data - simulate space elasticity
historical_data = []

for sku_id in range(1, 6):
    # Each SKU has different elasticity
    base_sales = np.random.uniform(50, 200)
    elasticity = np.random.uniform(0.2, 0.7)

    for period in range(20):
        facings = np.random.randint(2, 15)

        # Sales = base * (facings ^ elasticity) + noise
        sales = base_sales * (facings ** elasticity) + np.random.normal(0, 10)
        sales = max(0, sales)

        historical_data.append({
            'sku': f'SKU{sku_id:03d}',
            'period': period,
            'facings': facings,
            'sales_units': sales,
            'sales_dollars': sales * np.random.uniform(3, 8)
        })

historical_df = pd.DataFrame(historical_data)

# Analyze elasticity
analyzer = SpaceElasticityAnalyzer(historical_df)
elasticity = analyzer.calculate_space_elasticity('SKU001')

print(f"Space Elasticity: {elasticity['elasticity']:.3f}")
print(f"Interpretation: {elasticity['interpretation']}")
print(f"R-squared: {elasticity['r_squared']:.3f}")

# Find optimal facings
optimization = analyzer.find_optimal_facings(
    sku='SKU001',
    cost_per_facing=2.5,  # Cost per facing per week
    profit_per_unit=4.0,   # Profit margin per unit
    max_facings=15
)

print(f"\nOptimal facings: {optimization['optimal_facings']}")
print(f"Expected weekly sales: {optimization['expected_sales']:.0f} units")
print(f"Expected weekly profit: ${optimization['expected_profit']:.2f}")
```

---

## Planogram Optimization Models

### Fixture-Level Space Allocation

```python
class PlanogramOptimizer:
    """
    Optimize product placement and facings on a fixture

    Maximize sales/profit per square foot
    """

    def __init__(self, fixture_config, products_data):
        """
        Parameters:
        - fixture_config: Dict with fixture dimensions
          {'shelves': 5, 'width_inches': 48, 'depth_inches': 12}
        - products_data: DataFrame with product info
          columns: ['sku', 'width_inches', 'depth_inches', 'height_inches',
                   'weekly_sales', 'profit_per_unit', 'min_facings', 'max_facings']
        """
        self.fixture = fixture_config
        self.products = products_data

    def calculate_space_productivity(self, allocation):
        """
        Calculate sales and profit per square foot for an allocation

        allocation: Dict {sku: {'shelf': shelf_num, 'facings': count}}
        """

        total_sales = 0
        total_profit = 0
        space_used = {}  # Track space used per shelf

        for sku, placement in allocation.items():
            product = self.products[self.products['sku'] == sku].iloc[0]

            # Calculate space consumption
            facings = placement['facings']
            width_per_facing = product['width_inches']
            total_width = facings * width_per_facing

            shelf = placement['shelf']

            # Update space tracking
            if shelf not in space_used:
                space_used[shelf] = 0
            space_used[shelf] += total_width

            # Calculate sales (with space elasticity effect)
            base_sales = product['weekly_sales']
            elasticity = product.get('space_elasticity', 0.4)

            # Diminishing returns: sales = base * (facings ^ elasticity)
            adjusted_sales = base_sales * (facings ** elasticity)

            total_sales += adjusted_sales
            total_profit += adjusted_sales * product['profit_per_unit']

        # Calculate space utilization
        total_space_sqft = (
            self.fixture['shelves'] *
            self.fixture['width_inches'] *
            self.fixture['depth_inches'] / 144  # Convert to sqft
        )

        sales_per_sqft = total_sales / total_space_sqft if total_space_sqft > 0 else 0
        profit_per_sqft = total_profit / total_space_sqft if total_space_sqft > 0 else 0

        # Check constraints
        valid = True
        for shelf, width_used in space_used.items():
            if width_used > self.fixture['width_inches']:
                valid = False

        return {
            'total_sales': total_sales,
            'total_profit': total_profit,
            'sales_per_sqft': sales_per_sqft,
            'profit_per_sqft': profit_per_sqft,
            'space_utilization': space_used,
            'valid': valid
        }

    def greedy_allocation(self, objective='profit'):
        """
        Greedy algorithm to allocate products to fixture

        Prioritize by profit per square foot (or sales per sqft)
        """

        # Calculate priority score for each product
        self.products['priority'] = self.products.apply(
            lambda row: self._calculate_priority(row, objective),
            axis=1
        )

        # Sort by priority
        sorted_products = self.products.sort_values('priority', ascending=False)

        # Allocate
        allocation = {}
        shelf_space_remaining = {
            i: self.fixture['width_inches'] for i in range(self.fixture['shelves'])
        }

        for idx, product in sorted_products.iterrows():
            sku = product['sku']
            width = product['width_inches']
            min_facings = product.get('min_facings', 1)
            max_facings = product.get('max_facings', 10)

            # Try to allocate to best shelf position
            # Eye level (middle shelves) are most valuable
            shelf_priority = self._get_shelf_priority_order(self.fixture['shelves'])

            allocated = False

            for shelf in shelf_priority:
                max_facings_possible = int(shelf_space_remaining[shelf] / width)

                if max_facings_possible >= min_facings:
                    # Allocate optimal number of facings
                    facings = min(max_facings, max_facings_possible)

                    allocation[sku] = {
                        'shelf': shelf,
                        'facings': facings,
                        'position': 'center'  # Simplified
                    }

                    # Update remaining space
                    shelf_space_remaining[shelf] -= facings * width
                    allocated = True
                    break

            if not allocated:
                # Could not fit this product
                pass

        return allocation

    def _calculate_priority(self, product, objective):
        """Calculate priority score for product placement"""

        if objective == 'profit':
            # Profit per unit of space
            space_per_unit = product['width_inches'] * product['depth_inches']
            return (product['weekly_sales'] * product['profit_per_unit']) / space_per_unit

        else:  # sales
            space_per_unit = product['width_inches'] * product['depth_inches']
            return product['weekly_sales'] / space_per_unit

    def _get_shelf_priority_order(self, num_shelves):
        """
        Get shelf allocation priority

        Eye level (middle) is most valuable
        """

        if num_shelves <= 3:
            return list(range(num_shelves))

        # Middle shelves first
        middle = num_shelves // 2
        priority = [middle]

        # Alternate above and below middle
        for offset in range(1, num_shelves):
            if middle + offset < num_shelves:
                priority.append(middle + offset)
            if middle - offset >= 0:
                priority.append(middle - offset)

        return priority

    def optimize_with_constraints(self, objective='profit',
                                  category_constraints=None):
        """
        Optimize with business constraints

        Constraints:
        - Minimum facings per SKU
        - Maximum facings per SKU
        - Product grouping (keep related products together)
        - Brand requirements
        """

        # Use greedy as baseline
        allocation = self.greedy_allocation(objective)

        # Calculate performance
        performance = self.calculate_space_productivity(allocation)

        return allocation, performance

    def create_visual_planogram(self, allocation):
        """
        Create visual representation of planogram

        Returns ASCII art / simple visualization
        """

        # Group by shelf
        shelves = {}
        for sku, placement in allocation.items():
            shelf = placement['shelf']
            if shelf not in shelves:
                shelves[shelf] = []

            product = self.products[self.products['sku'] == sku].iloc[0]
            width = product['width_inches'] * placement['facings']

            shelves[shelf].append({
                'sku': sku,
                'facings': placement['facings'],
                'width': width
            })

        # Print planogram
        print(f"\nPLANOGRAM - {self.fixture['width_inches']}\" wide x {self.fixture['shelves']} shelves")
        print("=" * 60)

        for shelf in range(self.fixture['shelves'] - 1, -1, -1):  # Top to bottom
            print(f"Shelf {shelf + 1}:", end=" ")

            if shelf in shelves:
                for item in shelves[shelf]:
                    # Display SKU with facings
                    display = f"[{item['sku']}x{item['facings']}]"
                    print(display, end=" ")
            else:
                print("(empty)", end="")

            print()

        print("=" * 60)

# Example
fixture_config = {
    'shelves': 5,
    'width_inches': 48,
    'depth_inches': 12
}

products_data = pd.DataFrame({
    'sku': [f'SKU{i:03d}' for i in range(1, 16)],
    'width_inches': np.random.uniform(3, 8, 15),
    'depth_inches': np.random.uniform(4, 10, 15),
    'height_inches': np.random.uniform(6, 12, 15),
    'weekly_sales': np.random.uniform(10, 100, 15),
    'profit_per_unit': np.random.uniform(2, 8, 15),
    'space_elasticity': np.random.uniform(0.3, 0.6, 15),
    'min_facings': 1,
    'max_facings': np.random.randint(4, 12, 15)
})

optimizer = PlanogramOptimizer(fixture_config, products_data)

# Optimize
allocation, performance = optimizer.optimize_with_constraints(objective='profit')

print(f"Total weekly sales: ${performance['total_sales']:.0f}")
print(f"Total weekly profit: ${performance['total_profit']:.0f}")
print(f"Profit per sqft: ${performance['profit_per_sqft']:.2f}")

# Visualize
optimizer.create_visual_planogram(allocation)
```

---

## Category Management Integration

### Assortment-Space Optimization

```python
class CategorySpaceManager:
    """
    Manage category-level space allocation

    Decide how much space each category/subcategory gets
    """

    def __init__(self, store_data):
        self.store = store_data

    def allocate_space_to_categories(self, categories_data,
                                     total_space_sqft):
        """
        Allocate store space across categories

        Methods:
        - Sales-based: Proportional to sales
        - Profit-based: Proportional to profit
        - Hybrid: Balance sales and profit
        """

        # Calculate each category's contribution
        categories_data['sales_contribution'] = (
            categories_data['annual_sales'] /
            categories_data['annual_sales'].sum()
        )

        categories_data['profit_contribution'] = (
            categories_data['annual_profit'] /
            categories_data['annual_profit'].sum()
        )

        # Hybrid allocation (60% sales, 40% profit)
        categories_data['allocation_weight'] = (
            categories_data['sales_contribution'] * 0.6 +
            categories_data['profit_contribution'] * 0.4
        )

        # Allocate space
        categories_data['allocated_space_sqft'] = (
            categories_data['allocation_weight'] * total_space_sqft
        )

        # Calculate expected productivity
        categories_data['current_sales_per_sqft'] = (
            categories_data['annual_sales'] /
            categories_data['current_space_sqft']
        )

        categories_data['expected_sales_per_sqft'] = (
            categories_data['annual_sales'] /
            categories_data['allocated_space_sqft']
        )

        return categories_data[[
            'category', 'current_space_sqft', 'allocated_space_sqft',
            'annual_sales', 'annual_profit', 'current_sales_per_sqft',
            'expected_sales_per_sqft'
        ]]

    def recommend_space_adjustments(self, categories_data):
        """
        Identify categories that need more or less space

        Based on:
        - Sales per sqft vs. store average
        - Growth trends
        - Profit margins
        """

        store_avg_sales_per_sqft = (
            categories_data['annual_sales'].sum() /
            categories_data['current_space_sqft'].sum()
        )

        recommendations = []

        for idx, category in categories_data.iterrows():
            current_productivity = category['annual_sales'] / category['current_space_sqft']
            ratio_to_avg = current_productivity / store_avg_sales_per_sqft

            if ratio_to_avg > 1.3:
                recommendation = 'Expand space'
                reason = f"High productivity ({ratio_to_avg:.1f}x store average)"
                change_pct = '+15 to +25%'

            elif ratio_to_avg < 0.7:
                recommendation = 'Reduce space'
                reason = f"Low productivity ({ratio_to_avg:.1f}x store average)"
                change_pct = '-15 to -25%'

            else:
                recommendation = 'Maintain space'
                reason = 'Productivity in line with average'
                change_pct = '±5%'

            recommendations.append({
                'category': category['category'],
                'recommendation': recommendation,
                'reason': reason,
                'suggested_change': change_pct,
                'productivity_ratio': ratio_to_avg
            })

        return pd.DataFrame(recommendations)

# Example
categories_data = pd.DataFrame({
    'category': ['Dairy', 'Bakery', 'Produce', 'Meat', 'Frozen', 'Beverages'],
    'current_space_sqft': [800, 600, 1200, 900, 1500, 2000],
    'annual_sales': [1_200_000, 800_000, 1_500_000, 1_800_000, 1_400_000, 2_200_000],
    'annual_profit': [180_000, 240_000, 300_000, 360_000, 210_000, 330_000]
})

manager = CategorySpaceManager({})

# Allocate space
allocation = manager.allocate_space_to_categories(categories_data, total_space_sqft=7000)
print("Category Space Allocation:")
print(allocation)

# Recommendations
recommendations = manager.recommend_space_adjustments(categories_data)
print("\nSpace Adjustment Recommendations:")
print(recommendations)
```

---

## Cross-Merchandising & Adjacency

```python
class CrossMerchandisingOptimizer:
    """
    Optimize product adjacencies for cross-selling

    Place complementary products near each other
    """

    def __init__(self, products_data, affinity_matrix):
        """
        Parameters:
        - products_data: Product information
        - affinity_matrix: Cross-purchase patterns
          affinity_matrix[i][j] = likelihood customer buys j given they buy i
        """
        self.products = products_data
        self.affinity = affinity_matrix

    def identify_product_clusters(self, min_affinity=0.3):
        """
        Cluster products with high affinity

        Products that are frequently bought together
        """

        from sklearn.cluster import AgglomerativeClustering

        # Use affinity matrix for clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - min_affinity,
            affinity='precomputed',
            linkage='average'
        )

        # Convert affinity to distance
        distance_matrix = 1 - self.affinity
        clusters = clustering.fit_predict(distance_matrix)

        self.products['cluster'] = clusters

        return self.products[['sku', 'cluster']]

    def score_adjacency(self, sku1, sku2):
        """
        Score how beneficial it is to place two products adjacent

        Higher score = more beneficial
        """

        idx1 = self.products[self.products['sku'] == sku1].index[0]
        idx2 = self.products[self.products['sku'] == sku2].index[0]

        # Bidirectional affinity
        affinity_1_to_2 = self.affinity[idx1, idx2]
        affinity_2_to_1 = self.affinity[idx2, idx1]

        # Combined score
        adjacency_score = (affinity_1_to_2 + affinity_2_to_1) / 2

        return adjacency_score

    def recommend_adjacencies(self, sku, top_n=5):
        """
        Recommend products to place next to a given SKU

        Based on cross-purchase affinity
        """

        idx = self.products[self.products['sku'] == sku].index[0]

        # Get affinities for this product
        affinities = self.affinity[idx, :]

        # Sort and get top N
        top_indices = np.argsort(affinities)[::-1][:top_n + 1]  # +1 to exclude self

        recommendations = []
        for other_idx in top_indices:
            other_sku = self.products.iloc[other_idx]['sku']

            if other_sku == sku:
                continue  # Skip self

            recommendations.append({
                'sku': other_sku,
                'affinity_score': affinities[other_idx],
                'rationale': 'Frequently purchased together'
            })

        return recommendations[:top_n]

# Example
n_products = 20
products_data = pd.DataFrame({
    'sku': [f'SKU{i:03d}' for i in range(1, n_products + 1)]
})

# Generate sample affinity matrix
# (in practice, this comes from transaction data)
np.random.seed(42)
affinity_matrix = np.random.rand(n_products, n_products) * 0.5
np.fill_diagonal(affinity_matrix, 1.0)  # Product always purchased with itself

# Make it symmetric (for simplicity)
affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2

optimizer = CrossMerchandisingOptimizer(products_data, affinity_matrix)

# Identify clusters
clusters = optimizer.identify_product_clusters(min_affinity=0.3)
print("Product Clusters:")
print(clusters.groupby('cluster')['sku'].apply(list))

# Get adjacency recommendations
recommendations = optimizer.recommend_adjacencies('SKU001', top_n=5)
print(f"\nRecommended adjacencies for SKU001:")
for rec in recommendations:
    print(f"  {rec['sku']}: {rec['affinity_score']:.2f}")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- `scipy.optimize`: Non-linear optimization
- `pulp`, `pyomo`: Linear programming for space allocation
- `ortools`: Constraint programming for planograms

**Machine Learning:**
- `scikit-learn`: Clustering for product grouping
- `mlxtend`: Association rule mining (market basket analysis)

**Visualization:**
- `matplotlib`, `seaborn`: Planogram visualization
- `plotly`: Interactive layouts
- `PIL` (Pillow): Image-based planograms

### Commercial Software

**Planogram Software:**
- **JDA/Blue Yonder Intactix**: Enterprise space planning
- **RELEX Solutions**: Space & assortment optimization
- **Galleria by Movista**: Visual merchandising
- **Apollo by Shelf Logic**: AI-powered planograms
- **SCORPION by ESL**: Planogram automation

**Category Management:**
- **Nielsen Spaceman**: Space planning and optimization
- **IRI ProSpace**: Space productivity analytics
- **Symphony RetailAI**: AI-driven category management

**Specialized Tools:**
- **SmartDraw**: Basic planogram creation
- **PlanoHero**: Cloud planogram software
- **Quant**: Retail space intelligence

---

## Common Challenges & Solutions

### Challenge: Product Dimension Variability

**Problem:**
- Products have different sizes
- Irregular shapes don't fit neatly
- Wasted space from poor packing

**Solutions:**
- Modular shelf heights
- Adjustable dividers
- Product grouping by size
- Vertical stacking for small items
- Custom fixtures for odd shapes

### Challenge: Frequent Assortment Changes

**Problem:**
- New products introduced frequently
- Seasonal rotations
- Re-planogramming is labor-intensive

**Solutions:**
- Flexible planogram zones
- "Hot spot" areas for new products
- Micro-category approach (easier to swap)
- Digital planograms (easy updates)
- Planogram compliance automation

### Challenge: Store Format Diversity

**Problem:**
- Different store sizes
- Layout variations
- One planogram doesn't fit all

**Solutions:**
- Store clustering (A/B/C formats)
- Modular planogram approach
- Core vs. flex sections
- Automated planogram generation by store
- Local customization within guidelines

### Challenge: Operational Complexity

**Problem:**
- Restocking difficulty
- Labor time to execute resets
- Compliance monitoring hard

**Solutions:**
- Operational feasibility scoring
- Minimize SKU moves during resets
- Phased implementation
- Photo compliance apps
- Planogram simplification

### Challenge: Balancing Multiple Objectives

**Problem:**
- Maximize sales vs. profit
- Customer experience vs. efficiency
- Brand requirements vs. optimization
- Visual appeal vs. space productivity

**Solutions:**
- Multi-objective optimization
- Weighted scoring systems
- Constraints for brand/experience requirements
- A/B testing different approaches
- Category captain collaboration

---

## Output Format

### Planogram Optimization Report

**Executive Summary:**
- Category: Beverages (Soft Drinks section)
- Current performance: $450/sqft/week
- Optimized performance: $580/sqft/week (+29%)
- Fixture count: 12 fixtures (4ft sections)
- SKU count: 85 SKUs

**Current vs. Optimized Performance:**

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Sales per sqft per week | $450 | $580 | +29% |
| Profit per sqft per week | $95 | $135 | +42% |
| Space utilization | 78% | 94% | +16 pts |
| SKU count | 85 | 72 | -13 SKUs |
| Avg facings per SKU | 3.2 | 4.5 | +41% |

**Top Changes - SKU Level:**

| SKU | Product | Current Facings | Optimized Facings | Change | Rationale |
|-----|---------|----------------|-------------------|--------|-----------|
| SKU001 | Coke 12pk | 4 | 8 | +4 | High sales, elastic to space |
| SKU015 | Pepsi 2L | 6 | 4 | -2 | Low elasticity, over-spaced |
| SKU023 | LaCroix variety | 2 | 6 | +4 | Growing category, undersized |
| SKU045 | Generic cola | 3 | 0 | -3 (discontinue) | Poor sales per facing |

**Shelf-Level Plan:**

```
PLANOGRAM - Soft Drinks Section (48" x 5 shelves)
============================================================
Shelf 5: [Coke12pkx8] [Pepsi12pkx6] [Sprite12pkx5]
Shelf 4: [Coke2Lx6] [Pepsi2Lx4] [DrPepper2Lx4] [Sprite2Lx4]
Shelf 3: [LaCroixX6] [BublyX5] [Perrier6pkx4]
Shelf 2: [CokeCansx4] [PepsiCansx4] [Energy6pkx5]
Shelf 1: [2LSparkling1] [2LSparkling2] [Juice4pkx3]
============================================================
```

**Space Productivity by Fixture:**

| Fixture | Current $/sqft/wk | Optimized $/sqft/wk | Improvement |
|---------|------------------|---------------------|-------------|
| Fixture 1 (Eye-level) | $620 | $780 | +26% |
| Fixture 2 (Eye-level) | $580 | $750 | +29% |
| Fixture 3 (Chest-level) | $480 | $610 | +27% |
| Fixture 4 (Waist-level) | $380 | $490 | +29% |

**Cross-Merchandising Opportunities:**

1. **Chips + Dips cluster**: Add salsa adjacent to chips (+$8K annual sales)
2. **Pasta + Sauce**: Consolidate for convenience (+$12K annual sales)
3. **Baking needs**: Cluster flour, sugar, baking soda (+$6K annual sales)

**Implementation Plan:**

| Phase | Actions | SKUs Affected | Labor Hours | Timeline |
|-------|---------|---------------|-------------|----------|
| Phase 1 | Adjust facings (no moves) | 25 | 8 hours | Week 1 |
| Phase 2 | Relocate high-movers to eye-level | 15 | 12 hours | Week 2 |
| Phase 3 | Discontinue poor performers | 13 | 4 hours | Week 3 |
| Phase 4 | Final adjustments & cleanup | All | 6 hours | Week 4 |

**Risk & Mitigation:**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Out-of-stocks during reset | Medium | Medium | Overstock before reset, phased approach |
| Customer confusion | Low | High | Clear signage, staff briefing |
| Execution errors | Medium | Medium | Photo compliance, store visits |

---

## Questions to Ask

If you need more context:
1. What category/department needs optimization?
2. How many SKUs are in the category?
3. What's the current sales per square foot?
4. What fixtures are you using? (shelving, gondolas, end caps)
5. Do you have product dimension data?
6. Do you have historical sales by SKU?
7. Any space elasticity data? (testing different facings)
8. What are your constraints? (brand requirements, minimum facings)
9. Is this for one store or chain-wide?

---

## Related Skills

- **retail-allocation**: Initial inventory allocation to stores
- **retail-replenishment**: Restocking strategy
- **demand-forecasting**: Demand forecasting by SKU/store
- **inventory-optimization**: Safety stock and service levels
- **supply-chain-analytics**: Space productivity metrics
- **warehouse-slotting-optimization**: Similar concepts for warehouses

---
name: procurement-optimization
description: When the user wants to optimize procurement decisions, allocate orders across suppliers, or determine optimal order quantities. Also use when the user mentions "order allocation," "supplier portfolio optimization," "lot sizing," "order splitting," "purchase optimization," "EOQ," "sourcing optimization," or "multi-sourcing strategy." For supplier selection, see supplier-selection. For spend analysis, see spend-analysis.
---

# Procurement Optimization

You are an expert in procurement optimization and decision science. Your goal is to help organizations make optimal purchasing decisions that minimize total costs while meeting requirements for service levels, capacity constraints, and risk management.

## Initial Assessment

Before optimizing procurement, understand:

1. **Procurement Context**
   - What products/materials are being procured?
   - Current procurement process and pain points?
   - Spend volume and frequency?
   - Number of suppliers and current allocation?

2. **Business Objectives**
   - Primary goal? (cost, service, risk, sustainability)
   - Cost components? (price, freight, duties, carrying)
   - Service level requirements?
   - Risk tolerance?

3. **Constraints**
   - Supplier capacity limits?
   - Minimum order quantities (MOQs)?
   - Lead times and delivery windows?
   - Budget or working capital limits?
   - Quality or certification requirements?

4. **Data Availability**
   - Historical demand and order patterns?
   - Supplier pricing (including volume discounts)?
   - Inventory carrying costs?
   - Order processing costs?
   - Transportation and logistics costs?

---

## Procurement Optimization Framework

### Key Decision Areas

**1. Order Quantity Decisions**
- Economic Order Quantity (EOQ)
- Lot-sizing with constraints
- Quantity discount optimization
- Joint replenishment

**2. Supplier Allocation Decisions**
- Single vs. multi-sourcing
- Order splitting across suppliers
- Portfolio optimization
- Supplier diversification

**3. Timing Decisions**
- Reorder points
- Order scheduling
- Lead time management
- Safety stock levels

**4. Contract Decisions**
- Fixed vs. flexible quantities
- Price vs. volume commitments
- Long-term vs. spot buying
- Options and hedging

---

## Economic Order Quantity (EOQ)

### Classic EOQ Model

**Assumptions:**
- Constant demand rate
- Instantaneous replenishment
- No stockouts
- Fixed ordering cost and carrying cost

**Formula:**
```
EOQ = √(2 × D × S / H)

Where:
D = Annual demand (units)
S = Fixed ordering cost per order
H = Annual holding cost per unit
```

```python
import numpy as np
import matplotlib.pyplot as plt

def economic_order_quantity(annual_demand, order_cost, holding_cost_rate, unit_cost):
    """
    Calculate Economic Order Quantity

    Parameters:
    - annual_demand: units per year
    - order_cost: fixed cost per order ($)
    - holding_cost_rate: % of unit cost (e.g., 0.25 for 25%)
    - unit_cost: cost per unit ($)

    Returns:
    - EOQ, total annual cost, number of orders
    """

    holding_cost_per_unit = unit_cost * holding_cost_rate

    # EOQ formula
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)

    # Number of orders per year
    num_orders = annual_demand / eoq

    # Total annual cost
    ordering_cost = num_orders * order_cost
    holding_cost = (eoq / 2) * holding_cost_per_unit
    purchase_cost = annual_demand * unit_cost
    total_cost = ordering_cost + holding_cost + purchase_cost

    return {
        'eoq': round(eoq, 0),
        'num_orders_per_year': round(num_orders, 1),
        'order_frequency_days': round(365 / num_orders, 1),
        'total_annual_cost': round(total_cost, 2),
        'ordering_cost': round(ordering_cost, 2),
        'holding_cost': round(holding_cost, 2),
        'purchase_cost': round(purchase_cost, 2)
    }


# Example
result = economic_order_quantity(
    annual_demand=10000,      # 10,000 units/year
    order_cost=100,           # $100 per order
    holding_cost_rate=0.25,   # 25% carrying cost
    unit_cost=50              # $50/unit
)

print(f"Optimal Order Quantity: {result['eoq']} units")
print(f"Order Frequency: Every {result['order_frequency_days']} days")
print(f"Total Annual Cost: ${result['total_annual_cost']:,.0f}")
```

### EOQ with Quantity Discounts

**All-Units Discount:**
- Price break at certain volume levels
- All units purchased at discounted price

```python
def eoq_quantity_discounts(annual_demand, order_cost, holding_cost_rate, price_breaks):
    """
    EOQ with all-units quantity discounts

    price_breaks: list of (quantity, unit_price) tuples
    Example: [(0, 50), (500, 48), (1000, 46)]
    """

    price_breaks = sorted(price_breaks, key=lambda x: x[0])

    best_option = None
    best_cost = float('inf')

    for i, (min_qty, unit_price) in enumerate(price_breaks):
        holding_cost = unit_price * holding_cost_rate

        # Calculate EOQ for this price
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)

        # Determine feasible order quantity
        if i < len(price_breaks) - 1:
            max_qty = price_breaks[i + 1][0] - 1
        else:
            max_qty = float('inf')

        if eoq < min_qty:
            order_qty = min_qty  # Use minimum quantity for this price tier
        elif eoq > max_qty:
            continue  # EOQ not feasible in this tier
        else:
            order_qty = eoq

        # Calculate total cost
        num_orders = annual_demand / order_qty
        ordering_cost = num_orders * order_cost
        holding_cost_annual = (order_qty / 2) * holding_cost
        purchase_cost = annual_demand * unit_price
        total_cost = ordering_cost + holding_cost_annual + purchase_cost

        if total_cost < best_cost:
            best_cost = total_cost
            best_option = {
                'order_quantity': round(order_qty, 0),
                'unit_price': unit_price,
                'num_orders': round(num_orders, 1),
                'total_cost': round(total_cost, 2),
                'ordering_cost': round(ordering_cost, 2),
                'holding_cost': round(holding_cost_annual, 2),
                'purchase_cost': round(purchase_cost, 2)
            }

    return best_option


# Example with quantity discounts
price_breaks = [
    (0, 50),      # $50/unit for 0-499
    (500, 48),    # $48/unit for 500-999
    (1000, 46),   # $46/unit for 1000+
    (2000, 44)    # $44/unit for 2000+
]

result = eoq_quantity_discounts(
    annual_demand=10000,
    order_cost=100,
    holding_cost_rate=0.25,
    price_breaks=price_breaks
)

print(f"Optimal Order Quantity: {result['order_quantity']} units")
print(f"Unit Price: ${result['unit_price']}")
print(f"Total Annual Cost: ${result['total_cost']:,.0f}")
print(f"  Purchase: ${result['purchase_cost']:,.0f}")
print(f"  Ordering: ${result['ordering_cost']:,.0f}")
print(f"  Holding: ${result['holding_cost']:,.0f}")
```

---

## Supplier Allocation Optimization

### Multi-Sourcing Problem

**Objective:**
Allocate orders across multiple suppliers to minimize total cost while meeting capacity, quality, and risk constraints.

**Mathematical Formulation:**

```
Decision Variables:
  x_i = quantity ordered from supplier i

Objective:
  Minimize: Σ (p_i × x_i + f_i × y_i + t_i × x_i)

Where:
  p_i = unit price from supplier i
  f_i = fixed ordering cost from supplier i
  t_i = transportation cost per unit from supplier i
  y_i = binary (1 if order from supplier i, 0 otherwise)

Constraints:
  Σ x_i >= D                    (meet demand)
  x_i <= C_i × y_i              (supplier capacity)
  x_i >= MOQ_i × y_i            (minimum order quantity)
  Σ (q_i × x_i) / Σ x_i >= Q   (average quality requirement)
  x_i / Σ x_i <= R_max          (diversification - max % per supplier)
```

```python
from pulp import *
import pandas as pd

def optimize_supplier_allocation(suppliers, demand, constraints=None):
    """
    Optimize order allocation across multiple suppliers

    suppliers: DataFrame with columns:
        - supplier_id, unit_price, fixed_cost, capacity, moq,
          transport_cost, quality_score, lead_time

    demand: total quantity needed

    constraints: dict with optional keys:
        - min_quality: minimum average quality score
        - max_supplier_share: max % of demand from one supplier
        - max_suppliers: maximum number of suppliers to use
    """

    if constraints is None:
        constraints = {}

    # Create problem
    prob = LpProblem("Supplier_Allocation", LpMinimize)

    # Decision variables
    # x[i] = quantity from supplier i
    x = LpVariable.dicts("Quantity",
                        suppliers.index,
                        lowBound=0,
                        cat='Continuous')

    # y[i] = 1 if order from supplier i, 0 otherwise
    y = LpVariable.dicts("Use",
                        suppliers.index,
                        cat='Binary')

    # Objective function: minimize total cost
    prob += (
        # Variable costs (unit price + transport)
        lpSum([(suppliers.loc[i, 'unit_price'] +
                suppliers.loc[i, 'transport_cost']) * x[i]
               for i in suppliers.index]) +

        # Fixed ordering costs
        lpSum([suppliers.loc[i, 'fixed_cost'] * y[i]
               for i in suppliers.index])
    )

    # Constraint 1: Meet total demand
    prob += lpSum([x[i] for i in suppliers.index]) >= demand, "Meet_Demand"

    # Constraint 2: Capacity limits
    for i in suppliers.index:
        prob += x[i] <= suppliers.loc[i, 'capacity'] * y[i], f"Capacity_{i}"

    # Constraint 3: Minimum order quantities
    for i in suppliers.index:
        prob += x[i] >= suppliers.loc[i, 'moq'] * y[i], f"MOQ_{i}"

    # Constraint 4: Quality requirement
    if 'min_quality' in constraints:
        prob += (
            lpSum([suppliers.loc[i, 'quality_score'] * x[i]
                   for i in suppliers.index]) >=
            constraints['min_quality'] * demand,
            "Min_Quality"
        )

    # Constraint 5: Diversification (max share per supplier)
    if 'max_supplier_share' in constraints:
        for i in suppliers.index:
            prob += (
                x[i] <= constraints['max_supplier_share'] * demand,
                f"Max_Share_{i}"
            )

    # Constraint 6: Limit number of suppliers
    if 'max_suppliers' in constraints:
        prob += (
            lpSum([y[i] for i in suppliers.index]) <=
            constraints['max_suppliers'],
            "Max_Suppliers"
        )

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract results
    if LpStatus[prob.status] != 'Optimal':
        return {'status': LpStatus[prob.status], 'solution': None}

    results = []
    for i in suppliers.index:
        if x[i].varValue > 0.01:
            qty = x[i].varValue
            unit_cost = (suppliers.loc[i, 'unit_price'] +
                        suppliers.loc[i, 'transport_cost'])
            variable_cost = qty * unit_cost
            fixed_cost = suppliers.loc[i, 'fixed_cost']
            total_cost = variable_cost + fixed_cost

            results.append({
                'Supplier': suppliers.loc[i, 'supplier_id'],
                'Quantity': round(qty, 0),
                'Share_%': round(qty / demand * 100, 1),
                'Unit_Cost': unit_cost,
                'Variable_Cost': round(variable_cost, 2),
                'Fixed_Cost': fixed_cost,
                'Total_Cost': round(total_cost, 2),
                'Quality_Score': suppliers.loc[i, 'quality_score'],
                'Lead_Time': suppliers.loc[i, 'lead_time']
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total_Cost')

    return {
        'status': 'Optimal',
        'total_cost': round(value(prob.objective), 2),
        'allocation': results_df,
        'avg_quality': round(
            (results_df['Quantity'] * results_df['Quality_Score']).sum() /
            results_df['Quantity'].sum(), 2
        ),
        'num_suppliers': len(results_df)
    }


# Example usage
suppliers_data = pd.DataFrame({
    'supplier_id': ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D'],
    'unit_price': [10.0, 9.5, 10.5, 9.8],
    'fixed_cost': [500, 400, 600, 450],
    'transport_cost': [1.0, 1.5, 0.8, 1.2],
    'capacity': [5000, 4000, 6000, 3000],
    'moq': [500, 300, 400, 200],
    'quality_score': [9, 8, 10, 8.5],  # 0-10 scale
    'lead_time': [21, 28, 14, 21]  # days
})

result = optimize_supplier_allocation(
    suppliers=suppliers_data,
    demand=8000,
    constraints={
        'min_quality': 8.5,        # Minimum average quality
        'max_supplier_share': 0.6, # Max 60% from one supplier
        'max_suppliers': 3         # Use at most 3 suppliers
    }
)

print(f"Status: {result['status']}")
print(f"Total Cost: ${result['total_cost']:,.2f}")
print(f"Number of Suppliers: {result['num_suppliers']}")
print(f"Average Quality: {result['avg_quality']}/10")
print("\nAllocation:")
print(result['allocation'])
```

### Portfolio Optimization Approach

**Efficient Frontier:**
Trade-off between cost and risk (supplier diversification)

```python
import numpy as np
from scipy.optimize import minimize

def supplier_portfolio_optimization(suppliers_df, demand,
                                   risk_aversion=0.5):
    """
    Optimize supplier portfolio considering cost and risk

    suppliers_df: DataFrame with unit_cost, std_dev (cost volatility)
    risk_aversion: 0 = cost only, 1 = risk only, 0.5 = balanced
    """

    n_suppliers = len(suppliers_df)

    def objective(weights):
        """Minimize weighted combination of cost and risk"""

        # Expected total cost
        expected_cost = np.sum(
            weights * suppliers_df['unit_cost'].values * demand
        )

        # Portfolio risk (variance of cost)
        # Simplified: assumes independence
        cost_variance = np.sum(
            (weights * demand) ** 2 * suppliers_df['std_dev'].values ** 2
        )
        cost_risk = np.sqrt(cost_variance)

        # Combined objective
        return (1 - risk_aversion) * expected_cost + risk_aversion * cost_risk

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
    ]

    # Bounds: each weight between 0 and max_share
    bounds = [(0, 0.6) for _ in range(n_suppliers)]

    # Initial guess: equal weights
    x0 = np.ones(n_suppliers) / n_suppliers

    # Optimize
    result = minimize(objective, x0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    if result.success:
        weights = result.x
        allocation = weights * demand

        return {
            'weights': weights,
            'allocation': allocation,
            'expected_cost': np.sum(weights * suppliers_df['unit_cost'].values * demand),
            'cost_std': np.sqrt(np.sum((weights * demand) ** 2 *
                                      suppliers_df['std_dev'].values ** 2))
        }
    else:
        return None
```

---

## Advanced Procurement Models

### Joint Replenishment Problem (JRP)

**Multiple Items from Same Supplier:**
- Share fixed ordering cost
- Coordinate order timing
- Minimize total cost

```python
def joint_replenishment_problem(items_df, shared_fixed_cost):
    """
    Joint replenishment for multiple items

    items_df: DataFrame with annual_demand, unit_cost, holding_cost_rate
    shared_fixed_cost: fixed cost incurred per joint order
    """

    # Calculate individual EOQs
    items_df['individual_eoq'] = np.sqrt(
        (2 * items_df['annual_demand'] * shared_fixed_cost) /
        (items_df['unit_cost'] * items_df['holding_cost_rate'])
    )

    # Calculate individual order frequencies
    items_df['frequency'] = items_df['annual_demand'] / items_df['individual_eoq']

    # Basic power-of-two policy
    # Find base frequency (highest frequency)
    base_frequency = items_df['frequency'].max()

    # Assign each item to nearest power-of-2 multiple of base
    items_df['assigned_frequency'] = items_df['frequency'].apply(
        lambda f: base_frequency / (2 ** round(np.log2(base_frequency / f)))
    )

    items_df['order_quantity'] = (
        items_df['annual_demand'] / items_df['assigned_frequency']
    )

    # Calculate costs
    items_df['ordering_cost'] = (
        shared_fixed_cost * items_df['assigned_frequency'] / len(items_df)
    )

    items_df['holding_cost'] = (
        items_df['order_quantity'] / 2 *
        items_df['unit_cost'] *
        items_df['holding_cost_rate']
    )

    items_df['total_cost'] = (
        items_df['ordering_cost'] +
        items_df['holding_cost'] +
        items_df['annual_demand'] * items_df['unit_cost']
    )

    joint_order_frequency = base_frequency
    days_between_orders = 365 / joint_order_frequency

    return {
        'items': items_df,
        'joint_order_frequency': round(joint_order_frequency, 1),
        'days_between_orders': round(days_between_orders, 1),
        'total_annual_cost': round(items_df['total_cost'].sum(), 2)
    }


# Example: 3 items ordered together
items = pd.DataFrame({
    'item': ['Item_A', 'Item_B', 'Item_C'],
    'annual_demand': [5000, 3000, 8000],
    'unit_cost': [10, 25, 5],
    'holding_cost_rate': [0.25, 0.25, 0.25]
})

result = joint_replenishment_problem(items, shared_fixed_cost=200)

print(f"Joint Order Frequency: Every {result['days_between_orders']} days")
print(f"Total Annual Cost: ${result['total_annual_cost']:,.0f}")
print("\nItem Details:")
print(result['items'][['item', 'order_quantity', 'assigned_frequency']])
```

### Dynamic Lot Sizing (Wagner-Whitin)

**Time-Varying Demand:**
- Demand varies by period
- No backorders
- Minimize total cost over planning horizon

```python
def wagner_whitin(demands, setup_cost, holding_cost_per_unit):
    """
    Wagner-Whitin algorithm for dynamic lot sizing

    demands: list of demands by period [d1, d2, d3, ...]
    setup_cost: fixed cost per order
    holding_cost_per_unit: cost to hold 1 unit for 1 period

    Returns: optimal order quantities and total cost
    """

    n_periods = len(demands)

    # DP arrays
    cost = [float('inf')] * (n_periods + 1)
    cost[0] = 0
    order_in = [0] * (n_periods + 1)

    # Forward recursion
    for t in range(1, n_periods + 1):
        for s in range(0, t):
            # Order in period s+1 to cover periods s+1 through t
            cum_demand = sum(demands[s:t])

            # Holding cost: carry inventory forward
            hold_cost = sum(
                (t - k - 1) * demands[k] * holding_cost_per_unit
                for k in range(s, t)
            )

            total_cost = cost[s] + setup_cost + hold_cost

            if total_cost < cost[t]:
                cost[t] = total_cost
                order_in[t] = s + 1

    # Backtrack to find order quantities
    orders = [0] * n_periods
    period = n_periods

    while period > 0:
        order_period = order_in[period]
        order_qty = sum(demands[order_period - 1:period])
        orders[order_period - 1] = order_qty
        period = order_period - 1

    return {
        'order_quantities': orders,
        'total_cost': cost[n_periods],
        'num_orders': sum(1 for q in orders if q > 0)
    }


# Example: 6-period planning horizon
demands = [100, 150, 200, 80, 120, 180]  # Units per period
setup_cost = 500
holding_cost = 2  # $ per unit per period

result = wagner_whitin(demands, setup_cost, holding_cost)

print("Optimal Order Plan:")
for t, qty in enumerate(result['order_quantities'], 1):
    if qty > 0:
        print(f"  Period {t}: Order {qty} units")

print(f"\nTotal Cost: ${result['total_cost']:,.2f}")
print(f"Number of Orders: {result['num_orders']}")
```

---

## Procurement Risk Management

### Supply Risk Metrics

```python
def calculate_supply_risk_score(supplier_data):
    """
    Calculate comprehensive supply risk score

    supplier_data: dict with risk factors
    Returns: risk score (0-100, higher = riskier)
    """

    risk_score = 0
    factors = []

    # Concentration risk (% of spend with supplier)
    spend_concentration = supplier_data.get('spend_share', 0)
    if spend_concentration > 0.5:
        risk_score += 25
        factors.append("High spend concentration")
    elif spend_concentration > 0.3:
        risk_score += 15
        factors.append("Moderate spend concentration")

    # Geographic risk
    if supplier_data.get('single_location', False):
        risk_score += 15
        factors.append("Single location risk")

    if supplier_data.get('geopolitical_risk', False):
        risk_score += 20
        factors.append("Geopolitical risk")

    # Financial health (0-10 scale, 10 = best)
    financial_score = supplier_data.get('financial_health', 7)
    if financial_score < 5:
        risk_score += 20
        factors.append("Poor financial health")
    elif financial_score < 7:
        risk_score += 10
        factors.append("Moderate financial concerns")

    # Capacity utilization
    capacity_util = supplier_data.get('capacity_utilization', 0.7)
    if capacity_util > 0.95:
        risk_score += 15
        factors.append("Very high capacity utilization")
    elif capacity_util > 0.85:
        risk_score += 8
        factors.append("High capacity utilization")

    # Quality issues (defect rate)
    defect_rate = supplier_data.get('defect_rate_ppm', 0)
    if defect_rate > 1000:
        risk_score += 15
        factors.append("Quality issues")
    elif defect_rate > 500:
        risk_score += 8

    # Delivery performance
    otd_rate = supplier_data.get('on_time_delivery', 1.0)
    if otd_rate < 0.90:
        risk_score += 15
        factors.append("Poor delivery performance")
    elif otd_rate < 0.95:
        risk_score += 8

    risk_level = 'Low' if risk_score < 30 else 'Medium' if risk_score < 60 else 'High'

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': factors
    }
```

### Optimal Dual Sourcing

**Balance cost vs. risk:**

```python
def optimal_dual_sourcing(primary_supplier, backup_supplier,
                         annual_demand, disruption_prob, disruption_cost):
    """
    Determine optimal split between primary and backup supplier

    Primary supplier: lower cost, higher risk
    Backup supplier: higher cost, lower risk
    """

    best_split = None
    best_expected_cost = float('inf')

    # Try different splits from 100/0 to 50/50
    for primary_pct in range(50, 101, 5):
        backup_pct = 100 - primary_pct

        primary_qty = annual_demand * (primary_pct / 100)
        backup_qty = annual_demand * (backup_pct / 100)

        # Direct costs
        primary_cost = primary_qty * primary_supplier['unit_cost']
        backup_cost = backup_qty * backup_supplier['unit_cost']

        # Expected disruption cost
        # Assuming backup supplier not disrupted when primary is
        expected_disruption = (
            disruption_prob *
            (primary_pct / 100) *
            disruption_cost
        )

        total_expected_cost = primary_cost + backup_cost + expected_disruption

        if total_expected_cost < best_expected_cost:
            best_expected_cost = total_expected_cost
            best_split = {
                'primary_pct': primary_pct,
                'backup_pct': backup_pct,
                'primary_qty': round(primary_qty, 0),
                'backup_qty': round(backup_qty, 0),
                'primary_cost': round(primary_cost, 2),
                'backup_cost': round(backup_cost, 2),
                'expected_disruption_cost': round(expected_disruption, 2),
                'total_expected_cost': round(total_expected_cost, 2)
            }

    return best_split


# Example
primary = {'unit_cost': 10.0}
backup = {'unit_cost': 11.5}

result = optimal_dual_sourcing(
    primary_supplier=primary,
    backup_supplier=backup,
    annual_demand=10000,
    disruption_prob=0.15,  # 15% chance of disruption
    disruption_cost=500000  # $500K cost if disrupted
)

print(f"Optimal Split: {result['primary_pct']}% / {result['backup_pct']}%")
print(f"Primary Quantity: {result['primary_qty']:,.0f} units")
print(f"Backup Quantity: {result['backup_qty']:,.0f} units")
print(f"Total Expected Cost: ${result['total_expected_cost']:,.2f}")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- `pulp`: Linear programming (supplier allocation, lot sizing)
- `scipy.optimize`: General optimization (portfolio, dual sourcing)
- `pyomo`: Advanced optimization modeling
- `cvxpy`: Convex optimization
- `ortools`: Google OR-Tools (constraint programming)

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `statsmodels`: Statistical analysis

**Visualization:**
- `matplotlib`, `seaborn`: Charts and plots
- `plotly`: Interactive dashboards

### Commercial Software

**Procurement Optimization:**
- **SAP Ariba**: Strategic sourcing and procurement
- **Coupa**: Source-to-pay platform
- **Jaggaer**: Strategic sourcing suite
- **GEP SMART**: Unified procurement
- **PROS**: Price and profit optimization
- **Keelvar**: Sourcing optimization

**Supply Chain Optimization:**
- **LLamasoft**: Supply chain design and optimization
- **Blue Yonder**: Supply chain planning
- **o9 Solutions**: Integrated planning
- **Kinaxis RapidResponse**: S&OP platform

**Analytics:**
- **Tableau**, **Power BI**: Procurement dashboards
- **SpendHQ**: Spend analytics
- **Zycus**: Spend analysis

---

## Common Challenges & Solutions

### Challenge: Quantity Discount Complexity

**Problem:**
- Multiple price breaks
- Different discount structures per supplier
- Hard to compare apples-to-apples

**Solutions:**
- Use optimization to evaluate all combinations
- Calculate total landed cost including carrying
- Sensitivity analysis on demand uncertainty
- Consider cash flow impact of large orders

### Challenge: Minimum Order Quantities (MOQs)

**Problem:**
- MOQs create excess inventory
- May force use of non-optimal suppliers
- Conflicts with JIT goals

**Solutions:**
- Negotiate lower MOQs with volume commitments
- Joint orders with other business units
- Consolidate similar items
- Evaluate total cost including holding costs
- Use contract manufacturers or distributors

### Challenge: Lead Time Variability

**Problem:**
- Uncertain delivery times
- Impacts safety stock needs
- Complicates order timing

**Solutions:**
- Model lead time as probability distribution
- Optimize reorder points under uncertainty
- Diversify suppliers by geography
- Implement vendor-managed inventory (VMI)
- Use tracking and visibility tools

### Challenge: Multi-Objective Trade-offs

**Problem:**
- Conflicting goals (cost, risk, quality, sustainability)
- Different stakeholder priorities
- Hard to quantify some objectives

**Solutions:**
- Multi-criteria decision analysis (weighted scoring)
- Pareto optimization (efficient frontier)
- Scenario analysis showing trade-offs
- Stakeholder workshops to align priorities
- Set constraints on secondary objectives

### Challenge: Demand Uncertainty

**Problem:**
- Forecast errors lead to over/under ordering
- Optimal order quantity changes with demand
- Risk of obsolescence or stockouts

**Solutions:**
- Use expected demand in EOQ calculations
- Safety stock optimization
- Flexible contracts (options, postponement)
- Vendor-managed inventory (VMI)
- Periodic review and adjustment
- Risk pooling through postponement

---

## Output Format

### Procurement Optimization Report

**Executive Summary:**
- Recommended procurement strategy
- Total cost and savings opportunity
- Key changes from current approach
- Implementation requirements

**Optimal Order Allocation:**

| Supplier | Allocation | Share % | Unit Cost | Total Cost | Quality | Lead Time | Risk Level |
|----------|------------|---------|-----------|------------|---------|-----------|------------|
| Supplier B | 4,800 units | 60% | $11.00 | $52,800 | 9/10 | 21 days | Low |
| Supplier C | 2,400 units | 30% | $11.30 | $27,120 | 10/10 | 14 days | Low |
| Supplier D | 800 units | 10% | $11.00 | $8,800 | 8.5/10 | 21 days | Medium |
| **Total** | **8,000 units** | **100%** | **$11.09** | **$88,720** | **9.2/10** | **19 days** | **Low** |

**Cost Breakdown:**

| Component | Current | Optimized | Savings | % Change |
|-----------|---------|-----------|---------|----------|
| Purchase Price | $95,000 | $88,000 | $7,000 | -7.4% |
| Transportation | $12,000 | $10,400 | $1,600 | -13.3% |
| Ordering Costs | $2,400 | $1,350 | $1,050 | -43.8% |
| Holding Costs | $18,000 | $15,500 | $2,500 | -13.9% |
| **Total** | **$127,400** | **$115,250** | **$12,150** | **-9.5%** |

**Order Schedule:**

```
Recommended Order Plan (Next 12 Months):

Q1:
  - Order 1,200 units from Supplier B (Week 1)
  - Order 600 units from Supplier C (Week 1)
  - Order 200 units from Supplier D (Week 1)

Q2:
  - Order 1,200 units from Supplier B (Week 14)
  - Order 600 units from Supplier C (Week 14)

Q3:
  - Order 1,200 units from Supplier B (Week 27)
  - Order 600 units from Supplier C (Week 27)
  - Order 300 units from Supplier D (Week 27)

Q4:
  - Order 1,200 units from Supplier B (Week 40)
  - Order 600 units from Supplier C (Week 40)
  - Order 300 units from Supplier D (Week 40)
```

**Risk Assessment:**

- Overall supply risk: **Low**
- No single supplier >60% of volume (diversified)
- Average supplier financial health: 8.5/10 (strong)
- Geographic diversification: 3 regions
- Quality performance: 99.1% defect-free (excellent)

**Recommendations:**

1. Transition to 60/30/10 split across three suppliers
2. Implement quarterly orders to balance ordering and holding costs
3. Negotiate 2-year contracts with volume commitments for price stability
4. Establish performance KPIs and quarterly reviews
5. Maintain qualified backup supplier (Supplier A) for emergencies

**Implementation Plan:**

- Month 1: Finalize contracts with selected suppliers
- Month 2: Place initial orders and validate quality
- Month 3: Ramp to full production volumes
- Month 4+: Monitor performance and adjust as needed

---

## Questions to Ask

If you need more context:
1. What products/materials are being procured?
2. What's the annual demand volume and variability?
3. How many suppliers are available and what are their capabilities?
4. What are the key cost drivers? (unit price, transportation, holding)
5. Any constraints? (MOQs, capacity limits, quality requirements)
6. What's the current procurement approach and pain points?
7. What's more important: lowest cost, risk mitigation, or quality?
8. Are there quantity discounts or price breaks?
9. What lead times and delivery performance do suppliers offer?
10. Is this a one-time purchase or ongoing replenishment?

---

## Related Skills

- **supplier-selection**: For evaluating and selecting suppliers
- **strategic-sourcing**: For category strategy and sourcing approach
- **spend-analysis**: For analyzing spend patterns and opportunities
- **inventory-optimization**: For safety stock and reorder points
- **supplier-risk-management**: For monitoring supplier risks
- **contract-management**: For negotiating optimal contract terms
- **demand-forecasting**: For demand inputs to procurement planning
