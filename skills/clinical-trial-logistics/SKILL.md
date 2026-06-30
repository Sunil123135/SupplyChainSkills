---
name: clinical-trial-logistics
description: "When the user wants to optimize clinical trial supply chain, manage investigational products, implement IRT systems, or ensure GCP compliance. Also use when the user mentions \"clinical trial supply,\" \"IMP logistics,\" \"IVRS/IWRS,\" \"drug accountability,\" \"randomization and supply,\" \"comparator sourcing,\" \"depot management,\" \"clinical packaging,\" \"site resupply,\" or \"GCP compliance.\" For pharmacy operations, see pharmacy-supply-chain. For general healthcare logistics, see hospital-logistics."
---

# Clinical Trial Logistics

You are an expert in clinical trial supply chain management and logistics. Your goal is to ensure reliable, compliant supply of investigational medicinal products (IMPs) to clinical trial sites while maintaining product integrity, regulatory compliance, and study blinding.

## Initial Assessment

Before optimizing clinical trial logistics, understand:

1. **Trial Characteristics**
   - Trial phase? (Phase I, II, III, IV)
   - Number of sites and countries?
   - Patient enrollment targets and timeline?
   - Blinding requirements? (open-label, single-blind, double-blind)
   - Randomization complexity? (stratification factors)

2. **Product Requirements**
   - Drug form? (tablets, injectables, biologics)
   - Storage conditions? (room temp, refrigerated, frozen)
   - Stability and shelf life?
   - Comparator/placebo requirements?
   - Packaging configuration?

3. **Supply Chain Infrastructure**
   - IRT/IVRS/IWRS system in place?
   - Depot locations? (global, regional)
   - Direct-to-site vs. depot model?
   - Cold chain capabilities?
   - Backup supply strategy?

4. **Compliance & Regulations**
   - GCP (Good Clinical Practice) requirements?
   - Country-specific regulations?
   - Import/export licenses needed?
   - Temperature excursion protocols?
   - Audit readiness?

---

## Clinical Trial Supply Chain Framework

### Trial Supply Models

**1. Direct-to-Site (DTS)**
- Ship directly from manufacturing to sites
- Pros: Reduced handling, faster delivery
- Cons: No buffer stock, complex global logistics
- Best for: Small trials, stable products

**2. Depot-Based Distribution**
- Regional depots hold inventory
- Ship to sites from nearest depot
- Pros: Faster resupply, buffer stock, consolidation
- Cons: Additional handling, storage costs
- Best for: Large global trials

**3. Hybrid Model**
- Depot for some regions, DTS for others
- Optimize based on site density and logistics
- Best for: Multi-regional trials with varied infrastructure

### IRT/IVRS/IWRS System Design

**Interactive Response Technology (IRT):**
- Randomization engine
- Supply allocation and tracking
- Temperature monitoring integration
- Drug accountability
- Resupply triggers

**Core Functions:**

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import random

class TreatmentArm(Enum):
    INVESTIGATIONAL = "investigational"
    COMPARATOR = "comparator"
    PLACEBO = "placebo"

class PatientStatus(Enum):
    SCREENED = "screened"
    RANDOMIZED = "randomized"
    ON_TREATMENT = "on_treatment"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"

@dataclass
class StratificationFactor:
    """Stratification criteria for randomization"""
    factor_name: str
    value: str

@dataclass
class Patient:
    """Clinical trial patient"""
    patient_id: str
    site_id: str
    screening_date: datetime
    status: PatientStatus
    stratification_factors: List[StratificationFactor] = None
    treatment_arm: Optional[TreatmentArm] = None
    randomization_date: Optional[datetime] = None
    allocated_kits: List[str] = None

@dataclass
class DrugKit:
    """IMP drug kit"""
    kit_number: str
    treatment_arm: TreatmentArm
    lot_number: str
    expiry_date: datetime
    site_id: str
    status: str  # available, allocated, dispensed, returned, destroyed
    patient_id: Optional[str] = None
    dispensed_date: Optional[datetime] = None

class IRTSystem:
    """
    Interactive Response Technology for clinical trials
    """

    def __init__(self, trial_id, randomization_ratio, blinded=True):
        self.trial_id = trial_id
        self.randomization_ratio = randomization_ratio  # e.g., {'investigational': 2, 'comparator': 1}
        self.blinded = blinded
        self.patients = {}
        self.drug_kits = {}
        self.randomization_list = []
        self.site_inventory = {}

    def generate_randomization_list(self, total_patients, block_size=6,
                                    stratification_factors=None):
        """
        Generate randomization list with blocking and stratification

        Parameters:
        - total_patients: Total randomization codes to generate
        - block_size: Block size for randomization
        - stratification_factors: List of stratification combinations
        """

        if stratification_factors is None:
            stratification_factors = [None]  # No stratification

        randomization_list = []
        randomization_number = 1

        for strata in stratification_factors:
            num_patients_per_strata = total_patients // len(stratification_factors)

            # Generate treatment sequence based on ratio
            sequence = []
            for treatment, count in self.randomization_ratio.items():
                sequence.extend([TreatmentArm[treatment.upper()]] * count)

            # Generate blocks
            num_blocks = (num_patients_per_strata // block_size) + 1

            for block in range(num_blocks):
                # Shuffle within block
                random.shuffle(sequence)

                for treatment in sequence:
                    if len(randomization_list) >= total_patients:
                        break

                    randomization_list.append({
                        'randomization_number': f"RND-{randomization_number:05d}",
                        'treatment_arm': treatment,
                        'stratification': strata,
                        'block': block + 1
                    })

                    randomization_number += 1

        self.randomization_list = randomization_list[:total_patients]

        return self.randomization_list

    def randomize_patient(self, patient_id, site_id, stratification_values=None):
        """
        Randomize patient and allocate treatment

        Parameters:
        - patient_id: Patient identifier
        - site_id: Study site
        - stratification_values: Dict of stratification factor values
        """

        if patient_id in self.patients:
            raise ValueError(f"Patient {patient_id} already randomized")

        # Find next available randomization code for stratification
        # In real system, this would be from pre-generated randomization list
        available_codes = [
            code for code in self.randomization_list
            if not any(p['randomization_code']['randomization_number'] == code['randomization_number']
                      for p in self.patients.values() if 'randomization_code' in p)
        ]

        if not available_codes:
            raise ValueError("No randomization codes available")

        # Assign next code
        randomization_code = available_codes[0]

        # Create patient record
        patient = {
            'patient_id': patient_id,
            'site_id': site_id,
            'randomization_date': datetime.now(),
            'status': PatientStatus.RANDOMIZED,
            'randomization_code': randomization_code,
            'treatment_arm': randomization_code['treatment_arm'],
            'stratification_values': stratification_values,
            'allocated_kits': []
        }

        self.patients[patient_id] = patient

        # Allocate drug kit
        kit = self._allocate_kit(patient_id, site_id, randomization_code['treatment_arm'])

        if kit:
            patient['allocated_kits'].append(kit['kit_number'])

        return {
            'patient_id': patient_id,
            'randomization_number': randomization_code['randomization_number'],
            'kit_number': kit['kit_number'] if kit else None,
            'dispensing_instructions': self._get_dispensing_instructions()
        }

    def _allocate_kit(self, patient_id, site_id, treatment_arm):
        """
        Allocate drug kit to patient from site inventory
        """

        # Find available kits at site for treatment arm
        site_kits = [
            kit for kit_num, kit in self.drug_kits.items()
            if kit['site_id'] == site_id
            and kit['treatment_arm'] == treatment_arm
            and kit['status'] == 'available'
            and kit['expiry_date'] > datetime.now()
        ]

        if not site_kits:
            # Trigger resupply
            self._trigger_resupply(site_id, treatment_arm)
            return None

        # Allocate kit with earliest expiry (FEFO)
        site_kits.sort(key=lambda x: x['expiry_date'])
        kit = site_kits[0]

        kit['status'] = 'allocated'
        kit['patient_id'] = patient_id
        kit['allocated_date'] = datetime.now()

        return kit

    def dispense_kit(self, kit_number, patient_id, dispensed_by):
        """
        Record kit dispensing to patient
        """

        if kit_number not in self.drug_kits:
            raise ValueError(f"Kit {kit_number} not found")

        kit = self.drug_kits[kit_number]

        if kit['status'] != 'allocated':
            raise ValueError(f"Kit {kit_number} is not allocated (status: {kit['status']})")

        if kit['patient_id'] != patient_id:
            raise ValueError(f"Kit {kit_number} is allocated to different patient")

        kit['status'] = 'dispensed'
        kit['dispensed_date'] = datetime.now()
        kit['dispensed_by'] = dispensed_by

        # Update patient status
        if patient_id in self.patients:
            self.patients[patient_id]['status'] = PatientStatus.ON_TREATMENT

        return {
            'kit_number': kit_number,
            'patient_id': patient_id,
            'dispensed_date': kit['dispensed_date'],
            'accountability_required': True
        }

    def return_kit(self, kit_number, return_reason, returned_by):
        """
        Record kit return (unused or partially used)
        """

        if kit_number not in self.drug_kits:
            raise ValueError(f"Kit {kit_number} not found")

        kit = self.drug_kits[kit_number]

        kit['status'] = 'returned'
        kit['return_date'] = datetime.now()
        kit['return_reason'] = return_reason
        kit['returned_by'] = returned_by

        return kit

    def check_site_inventory(self, site_id):
        """
        Check site inventory levels and trigger resupply if needed
        """

        site_kits = [
            kit for kit in self.drug_kits.values()
            if kit['site_id'] == site_id and kit['status'] == 'available'
        ]

        # Group by treatment arm
        inventory_by_arm = {}
        for arm in TreatmentArm:
            arm_kits = [k for k in site_kits if k['treatment_arm'] == arm]
            inventory_by_arm[arm.value] = {
                'available_kits': len(arm_kits),
                'expiring_soon': len([k for k in arm_kits if k['expiry_date'] < datetime.now() + timedelta(days=90)])
            }

        return inventory_by_arm

    def _trigger_resupply(self, site_id, treatment_arm):
        """
        Trigger site resupply when inventory low
        """

        resupply_request = {
            'site_id': site_id,
            'treatment_arm': treatment_arm,
            'request_date': datetime.now(),
            'priority': 'high',
            'requested_quantity': 20  # Standard resupply quantity
        }

        # In real system, this would integrate with depot management
        print(f"RESUPPLY TRIGGERED: Site {site_id} needs {treatment_arm.value} kits")

        return resupply_request

    def _get_dispensing_instructions(self):
        """
        Generate dispensing instructions for site staff
        """

        if self.blinded:
            return "Dispense assigned kit to patient. DO NOT OPEN OR INSPECT CONTENTS."
        else:
            return "Dispense assigned kit to patient. Verify drug name and strength."

# Example usage
irt = IRTSystem(
    trial_id='TRIAL-2024-001',
    randomization_ratio={'investigational': 2, 'comparator': 1, 'placebo': 1},
    blinded=True
)

# Generate randomization list
random.seed(42)
rand_list = irt.generate_randomization_list(
    total_patients=100,
    block_size=8
)

print(f"Generated {len(rand_list)} randomization codes")

# Add drug kits to site inventory
for i in range(20):
    kit_num = f"KIT-001-{1000+i}"
    arm = random.choice(list(TreatmentArm))

    irt.drug_kits[kit_num] = {
        'kit_number': kit_num,
        'treatment_arm': arm,
        'lot_number': 'LOT-2024-A',
        'expiry_date': datetime.now() + timedelta(days=730),
        'site_id': 'SITE-001',
        'status': 'available',
        'patient_id': None
    }

# Randomize patient
randomization = irt.randomize_patient(
    patient_id='PT-001-001',
    site_id='SITE-001',
    stratification_values={'age_group': '>=65', 'disease_severity': 'moderate'}
)

print(f"\nPatient randomized:")
print(f"  Randomization Number: {randomization['randomization_number']}")
print(f"  Kit Number: {randomization['kit_number']}")

# Dispense kit
dispense = irt.dispense_kit(
    kit_number=randomization['kit_number'],
    patient_id='PT-001-001',
    dispensed_by='Investigator Dr. Smith'
)

print(f"\nKit dispensed: {dispense['kit_number']} on {dispense['dispensed_date']}")

# Check site inventory
inventory = irt.check_site_inventory('SITE-001')
print(f"\nSite inventory:")
for arm, counts in inventory.items():
    print(f"  {arm}: {counts['available_kits']} kits available")
```

---

## Drug Accountability & Reconciliation

### Accountability Requirements

**GCP Requirements:**
- Receipt records
- Dispensing records
- Return records
- Destruction records
- Complete audit trail

```python
import pandas as pd
from datetime import datetime

class DrugAccountabilitySystem:
    """
    Manage drug accountability and reconciliation for clinical trials
    """

    def __init__(self, site_id, trial_id):
        self.site_id = site_id
        self.trial_id = trial_id
        self.transactions = []
        self.inventory = {}

    def receive_shipment(self, shipment_id, kits, received_by,
                        condition, temperature_log=None):
        """
        Record receipt of IMP shipment at site
        """

        receipt_transaction = {
            'transaction_type': 'receipt',
            'transaction_date': datetime.now(),
            'shipment_id': shipment_id,
            'received_by': received_by,
            'condition': condition,
            'temperature_compliant': self._verify_temperature(temperature_log),
            'kits': kits
        }

        self.transactions.append(receipt_transaction)

        # Add to inventory
        for kit in kits:
            self.inventory[kit['kit_number']] = {
                'kit_number': kit['kit_number'],
                'lot_number': kit['lot_number'],
                'expiry_date': kit['expiry_date'],
                'status': 'available',
                'received_date': datetime.now(),
                'patient_id': None
            }

        return receipt_transaction

    def dispense_to_patient(self, kit_number, patient_id, visit_number,
                           dispensed_by, dispense_date=None):
        """
        Record kit dispensing to patient
        """

        if kit_number not in self.inventory:
            raise ValueError(f"Kit {kit_number} not in inventory")

        if self.inventory[kit_number]['status'] != 'available':
            raise ValueError(f"Kit {kit_number} is not available")

        dispense_transaction = {
            'transaction_type': 'dispensed',
            'transaction_date': dispense_date or datetime.now(),
            'kit_number': kit_number,
            'patient_id': patient_id,
            'visit_number': visit_number,
            'dispensed_by': dispensed_by
        }

        self.transactions.append(dispense_transaction)

        # Update inventory
        self.inventory[kit_number]['status'] = 'dispensed'
        self.inventory[kit_number]['patient_id'] = patient_id
        self.inventory[kit_number]['dispensed_date'] = dispense_date or datetime.now()

        return dispense_transaction

    def return_from_patient(self, kit_number, patient_id, return_date,
                           units_returned, units_used, returned_by):
        """
        Record kit return from patient (compliance check)
        """

        if kit_number not in self.inventory:
            raise ValueError(f"Kit {kit_number} not in inventory")

        return_transaction = {
            'transaction_type': 'returned_from_patient',
            'transaction_date': return_date or datetime.now(),
            'kit_number': kit_number,
            'patient_id': patient_id,
            'units_returned': units_returned,
            'units_used': units_used,
            'compliance_pct': (units_used / (units_used + units_returned) * 100) if (units_used + units_returned) > 0 else 0,
            'returned_by': returned_by
        }

        self.transactions.append(return_transaction)

        self.inventory[kit_number]['status'] = 'returned_from_patient'
        self.inventory[kit_number]['units_returned'] = units_returned
        self.inventory[kit_number]['units_used'] = units_used

        return return_transaction

    def quarantine_kit(self, kit_number, reason, quarantined_by):
        """
        Quarantine kit (temperature excursion, damaged, etc.)
        """

        if kit_number not in self.inventory:
            raise ValueError(f"Kit {kit_number} not in inventory")

        quarantine_transaction = {
            'transaction_type': 'quarantined',
            'transaction_date': datetime.now(),
            'kit_number': kit_number,
            'reason': reason,
            'quarantined_by': quarantined_by
        }

        self.transactions.append(quarantine_transaction)

        self.inventory[kit_number]['status'] = 'quarantined'
        self.inventory[kit_number]['quarantine_reason'] = reason

        return quarantine_transaction

    def destroy_kit(self, kit_number, destruction_method, witnessed_by,
                   destruction_certificate=None):
        """
        Record kit destruction
        """

        if kit_number not in self.inventory:
            raise ValueError(f"Kit {kit_number} not in inventory")

        destruction_transaction = {
            'transaction_type': 'destroyed',
            'transaction_date': datetime.now(),
            'kit_number': kit_number,
            'destruction_method': destruction_method,
            'witnessed_by': witnessed_by,
            'destruction_certificate': destruction_certificate
        }

        self.transactions.append(destruction_transaction)

        self.inventory[kit_number]['status'] = 'destroyed'
        self.inventory[kit_number]['destruction_date'] = datetime.now()

        return destruction_transaction

    def return_to_sponsor(self, kit_numbers, shipment_id, shipped_by,
                         tracking_number, return_reason):
        """
        Return kits to sponsor/depot
        """

        return_transaction = {
            'transaction_type': 'returned_to_sponsor',
            'transaction_date': datetime.now(),
            'kit_numbers': kit_numbers,
            'shipment_id': shipment_id,
            'shipped_by': shipped_by,
            'tracking_number': tracking_number,
            'return_reason': return_reason
        }

        self.transactions.append(return_transaction)

        for kit_number in kit_numbers:
            if kit_number in self.inventory:
                self.inventory[kit_number]['status'] = 'returned_to_sponsor'
                self.inventory[kit_number]['return_date'] = datetime.now()

        return return_transaction

    def reconcile_inventory(self, physical_count_by_kit):
        """
        Reconcile physical inventory with system records
        """

        discrepancies = []

        for kit_number, physical_status in physical_count_by_kit.items():
            system_status = self.inventory.get(kit_number, {}).get('status', 'NOT_IN_SYSTEM')

            if physical_status != system_status:
                discrepancies.append({
                    'kit_number': kit_number,
                    'system_status': system_status,
                    'physical_status': physical_status,
                    'discrepancy_type': self._classify_discrepancy(system_status, physical_status)
                })

        reconciliation = {
            'reconciliation_date': datetime.now(),
            'site_id': self.site_id,
            'total_kits_system': len(self.inventory),
            'total_kits_physical': len(physical_count_by_kit),
            'discrepancies': discrepancies,
            'reconciliation_status': 'CLEAN' if len(discrepancies) == 0 else 'DISCREPANCIES_FOUND'
        }

        return reconciliation

    def _classify_discrepancy(self, system_status, physical_status):
        """Classify type of discrepancy"""
        if system_status == 'NOT_IN_SYSTEM':
            return 'EXTRA_KIT_FOUND'
        elif physical_status == 'NOT_FOUND':
            return 'KIT_MISSING'
        else:
            return 'STATUS_MISMATCH'

    def _verify_temperature(self, temperature_log):
        """Verify temperature compliance during shipment"""
        if not temperature_log:
            return None

        # Check all readings in range
        compliant = all(
            log['min_temp'] <= log['temperature'] <= log['max_temp']
            for log in temperature_log
        )

        return compliant

    def accountability_report(self, report_date=None):
        """
        Generate drug accountability report
        """

        report_date = report_date or datetime.now()

        # Group inventory by status
        inventory_df = pd.DataFrame(self.inventory.values())

        if len(inventory_df) == 0:
            return None

        status_summary = inventory_df.groupby('status').size().to_dict()

        # Recent transactions
        recent_transactions = [
            t for t in self.transactions
            if t['transaction_date'] >= report_date - timedelta(days=30)
        ]

        report = {
            'site_id': self.site_id,
            'trial_id': self.trial_id,
            'report_date': report_date,
            'inventory_summary': status_summary,
            'total_kits': len(inventory_df),
            'dispensed_kits': len(inventory_df[inventory_df['status'] == 'dispensed']),
            'available_kits': len(inventory_df[inventory_df['status'] == 'available']),
            'recent_transactions': len(recent_transactions),
            'transactions': recent_transactions
        }

        return report

# Example usage
accountability = DrugAccountabilitySystem(
    site_id='SITE-001',
    trial_id='TRIAL-2024-001'
)

# Receive shipment
kits = [
    {'kit_number': 'KIT-001-1001', 'lot_number': 'LOT-A', 'expiry_date': datetime(2026, 12, 31)},
    {'kit_number': 'KIT-001-1002', 'lot_number': 'LOT-A', 'expiry_date': datetime(2026, 12, 31)},
    {'kit_number': 'KIT-001-1003', 'lot_number': 'LOT-A', 'expiry_date': datetime(2026, 12, 31)}
]

receipt = accountability.receive_shipment(
    shipment_id='SHIP-2024-0015',
    kits=kits,
    received_by='Study Coordinator Jane Doe',
    condition='Good',
    temperature_log=[{'temperature': 6, 'min_temp': 2, 'max_temp': 8}]
)

print(f"Received {len(kits)} kits")

# Dispense to patient
dispense = accountability.dispense_to_patient(
    kit_number='KIT-001-1001',
    patient_id='PT-001-001',
    visit_number='Visit 2',
    dispensed_by='Investigator Dr. Smith'
)

print(f"Dispensed kit {dispense['kit_number']} to patient {dispense['patient_id']}")

# Generate accountability report
report = accountability.accountability_report()
print(f"\nAccountability Report:")
print(f"  Total kits: {report['total_kits']}")
print(f"  Available: {report['available_kits']}")
print(f"  Dispensed: {report['dispensed_kits']}")
print(f"  Inventory summary: {report['inventory_summary']}")
```

---

## Temperature-Controlled Logistics

### Cold Chain Management for Clinical Trials

**Temperature Ranges:**
- Room temperature: 15-25°C (59-77°F)
- Refrigerated: 2-8°C (36-46°F)
- Frozen: -20°C (-4°F)
- Ultra-cold: -80°C (-112°F)

```python
class ClinicalColdChainManager:
    """
    Manage temperature-controlled shipments for clinical trials
    """

    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.shipments = {}
        self.excursions = []

    def create_shipment(self, shipment_id, product_name, from_location,
                       to_location, temp_requirement, packaging_type):
        """
        Create temperature-controlled shipment
        """

        shipment = {
            'shipment_id': shipment_id,
            'product_name': product_name,
            'from_location': from_location,
            'to_location': to_location,
            'temp_requirement': temp_requirement,  # e.g., (2, 8) for 2-8°C
            'packaging_type': packaging_type,
            'ship_date': None,
            'delivery_date': None,
            'temperature_log': [],
            'excursions_detected': [],
            'status': 'pending'
        }

        self.shipments[shipment_id] = shipment

        return shipment

    def ship(self, shipment_id, data_logger_id, carrier, tracking_number):
        """
        Ship temperature-controlled package
        """

        if shipment_id not in self.shipments:
            raise ValueError(f"Shipment {shipment_id} not found")

        shipment = self.shipments[shipment_id]

        shipment['ship_date'] = datetime.now()
        shipment['data_logger_id'] = data_logger_id
        shipment['carrier'] = carrier
        shipment['tracking_number'] = tracking_number
        shipment['status'] = 'in_transit'

        return shipment

    def record_temperature(self, shipment_id, timestamp, temperature, location='in transit'):
        """
        Record temperature reading from data logger
        """

        if shipment_id not in self.shipments:
            raise ValueError(f"Shipment {shipment_id} not found")

        shipment = self.shipments[shipment_id]
        min_temp, max_temp = shipment['temp_requirement']

        reading = {
            'timestamp': timestamp,
            'temperature': temperature,
            'location': location,
            'in_range': min_temp <= temperature <= max_temp
        }

        shipment['temperature_log'].append(reading)

        # Check for excursion
        if not reading['in_range']:
            excursion = {
                'shipment_id': shipment_id,
                'timestamp': timestamp,
                'temperature': temperature,
                'required_range': shipment['temp_requirement'],
                'deviation': abs(temperature - ((min_temp + max_temp) / 2)),
                'location': location
            }

            shipment['excursions_detected'].append(excursion)
            self.excursions.append(excursion)

            # Trigger alert
            self._temperature_excursion_alert(excursion)

        return reading

    def deliver_shipment(self, shipment_id, received_by, condition_assessment):
        """
        Record shipment delivery
        """

        if shipment_id not in self.shipments:
            raise ValueError(f"Shipment {shipment_id} not found")

        shipment = self.shipments[shipment_id]

        shipment['delivery_date'] = datetime.now()
        shipment['received_by'] = received_by
        shipment['condition_assessment'] = condition_assessment
        shipment['status'] = 'delivered'

        # Analyze temperature compliance
        compliance = self._analyze_temperature_compliance(shipment)

        shipment['temperature_compliance'] = compliance

        return {
            'shipment_id': shipment_id,
            'delivery_date': shipment['delivery_date'],
            'temperature_compliant': compliance['compliant'],
            'excursions': len(shipment['excursions_detected']),
            'disposition': self._determine_disposition(compliance)
        }

    def _analyze_temperature_compliance(self, shipment):
        """
        Analyze temperature compliance for shipment
        """

        temp_log = shipment['temperature_log']

        if not temp_log:
            return {'compliant': None, 'reason': 'No temperature data'}

        total_readings = len(temp_log)
        compliant_readings = sum(1 for r in temp_log if r['in_range'])
        compliance_rate = (compliant_readings / total_readings * 100) if total_readings > 0 else 0

        num_excursions = len(shipment['excursions_detected'])

        # Determine compliance
        compliant = (compliance_rate >= 95 and num_excursions == 0)

        return {
            'compliant': compliant,
            'compliance_rate': round(compliance_rate, 2),
            'num_excursions': num_excursions,
            'total_readings': total_readings,
            'compliant_readings': compliant_readings
        }

    def _determine_disposition(self, compliance):
        """
        Determine product disposition based on compliance
        """

        if compliance['compliant']:
            return 'ACCEPT - Use per protocol'
        elif compliance['num_excursions'] > 0:
            return 'QUARANTINE - Investigate excursion, stability assessment required'
        else:
            return 'QUARANTINE - Temperature compliance review required'

    def _temperature_excursion_alert(self, excursion):
        """
        Send alert for temperature excursion
        """

        print(f"⚠ TEMPERATURE EXCURSION ALERT")
        print(f"  Shipment: {excursion['shipment_id']}")
        print(f"  Temperature: {excursion['temperature']}°C")
        print(f"  Required range: {excursion['required_range']}")
        print(f"  Time: {excursion['timestamp']}")

    def excursion_investigation_report(self, shipment_id):
        """
        Generate excursion investigation report
        """

        if shipment_id not in self.shipments:
            raise ValueError(f"Shipment {shipment_id} not found")

        shipment = self.shipments[shipment_id]

        if not shipment['excursions_detected']:
            return {'investigation_required': False}

        # Analyze excursions
        excursions_df = pd.DataFrame(shipment['excursions_detected'])

        report = {
            'shipment_id': shipment_id,
            'product_name': shipment['product_name'],
            'investigation_required': True,
            'num_excursions': len(shipment['excursions_detected']),
            'excursion_details': excursions_df.to_dict('records'),
            'total_time_out_of_range': 'Calculate from timestamp data',
            'max_deviation': excursions_df['deviation'].max() if len(excursions_df) > 0 else 0,
            'recommended_action': self._excursion_recommended_action(shipment),
            'stability_data_required': True,
            'sponsor_notification_required': True
        }

        return report

    def _excursion_recommended_action(self, shipment):
        """
        Recommend action for excursion
        """

        excursions = shipment['excursions_detected']

        if not excursions:
            return 'No action required'

        max_deviation = max(e['deviation'] for e in excursions)

        if max_deviation > 10:
            return 'REJECT - Significant excursion, product unusable'
        elif max_deviation > 5:
            return 'QUARANTINE - Contact sponsor, stability data review required'
        else:
            return 'QUARANTINE - Minor excursion, sponsor review required'

# Example usage
cold_chain = ClinicalColdChainManager(trial_id='TRIAL-2024-001')

# Create shipment
shipment = cold_chain.create_shipment(
    shipment_id='SHIP-2024-0020',
    product_name='Investigational Biologic ABC-123',
    from_location='Depot - Amsterdam',
    to_location='Site 105 - Memorial Hospital',
    temp_requirement=(2, 8),  # 2-8°C
    packaging_type='Qualified shipper with dry ice'
)

# Ship
cold_chain.ship(
    shipment_id='SHIP-2024-0020',
    data_logger_id='LOGGER-5678',
    carrier='FedEx Priority Overnight',
    tracking_number='FX123456789'
)

# Simulate temperature readings
import numpy as np
np.random.seed(42)

for hour in range(36):  # 36 hours in transit
    temp = 5 + np.random.normal(0, 1.5)

    # Simulate excursion at hour 20
    if hour == 20:
        temp = 12  # Excursion

    cold_chain.record_temperature(
        shipment_id='SHIP-2024-0020',
        timestamp=datetime.now() + timedelta(hours=hour),
        temperature=round(temp, 1)
    )

# Deliver
delivery = cold_chain.deliver_shipment(
    shipment_id='SHIP-2024-0020',
    received_by='Study Coordinator at Site 105',
    condition_assessment='Package intact, data logger attached'
)

print(f"Shipment delivered:")
print(f"  Temperature compliant: {delivery['temperature_compliant']}")
print(f"  Excursions: {delivery['excursions']}")
print(f"  Disposition: {delivery['disposition']}")

# Excursion investigation
if delivery['excursions'] > 0:
    investigation = cold_chain.excursion_investigation_report('SHIP-2024-0020')
    print(f"\n Excursion Investigation Required:")
    print(f"  Number of excursions: {investigation['num_excursions']}")
    print(f"  Max deviation: {investigation['max_deviation']:.1f}°C")
    print(f"  Recommended action: {investigation['recommended_action']}")
```

---

## Comparator Sourcing & Management

### Commercial Comparator Procurement

**Challenges:**
- Sourcing commercial drugs in different countries
- Ensuring consistent quality across batches
- Managing expiry dates
- Blinding/overencapsulation
- Import/export compliance

```python
class ComparatorManager:
    """
    Manage comparator drug sourcing and inventory
    """

    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.comparators = {}
        self.procurement_orders = []

    def add_comparator(self, comparator_id, drug_name, strength,
                      countries_needed, blinding_required):
        """
        Add comparator to trial requirements
        """

        self.comparators[comparator_id] = {
            'comparator_id': comparator_id,
            'drug_name': drug_name,
            'strength': strength,
            'countries_needed': countries_needed,
            'blinding_required': blinding_required,
            'sourcing_strategy': self._determine_sourcing_strategy(countries_needed)
        }

    def _determine_sourcing_strategy(self, countries_needed):
        """
        Determine optimal sourcing strategy
        """

        if len(countries_needed) == 1:
            return 'local_sourcing'
        elif len(countries_needed) <= 5:
            return 'regional_sourcing'
        else:
            return 'global_sourcing_multiple_suppliers'

    def create_procurement_order(self, comparator_id, country, quantity,
                                target_delivery_date, supplier=None):
        """
        Create procurement order for comparator
        """

        if comparator_id not in self.comparators:
            raise ValueError(f"Comparator {comparator_id} not defined")

        comparator = self.comparators[comparator_id]

        order = {
            'order_id': f"PO-{len(self.procurement_orders)+1:05d}",
            'comparator_id': comparator_id,
            'drug_name': comparator['drug_name'],
            'country': country,
            'quantity': quantity,
            'target_delivery_date': target_delivery_date,
            'supplier': supplier or 'To be determined',
            'order_date': datetime.now(),
            'status': 'pending',
            'import_license_required': self._check_import_requirements(country),
            'blinding_required': comparator['blinding_required']
        }

        self.procurement_orders.append(order)

        return order

    def _check_import_requirements(self, country):
        """Check if import license required"""
        # Simplified - would check regulatory database
        controlled_countries = ['US', 'CA', 'AU', 'JP']
        return country in controlled_countries

    def quality_assessment(self, order_id, batch_number, test_results):
        """
        Record quality assessment of received comparator
        """

        order = next((o for o in self.procurement_orders if o['order_id'] == order_id), None)

        if not order:
            raise ValueError(f"Order {order_id} not found")

        assessment = {
            'order_id': order_id,
            'batch_number': batch_number,
            'assessment_date': datetime.now(),
            'test_results': test_results,
            'acceptable': all(t['result'] == 'pass' for t in test_results),
            'release_status': 'released' if all(t['result'] == 'pass' for t in test_results) else 'rejected'
        }

        order['quality_assessment'] = assessment
        order['status'] = assessment['release_status']

        return assessment

# Example usage
comparator_mgr = ComparatorManager(trial_id='TRIAL-2024-001')

# Add comparator requirement
comparator_mgr.add_comparator(
    comparator_id='COMP-001',
    drug_name='Lipitor (Atorvastatin) 40mg',
    strength='40mg',
    countries_needed=['US', 'UK', 'Germany', 'France', 'Japan'],
    blinding_required=True
)

# Create procurement orders
order_us = comparator_mgr.create_procurement_order(
    comparator_id='COMP-001',
    country='US',
    quantity=5000,
    target_delivery_date=datetime.now() + timedelta(days=90),
    supplier='Cardinal Health'
)

print(f"Procurement order created: {order_us['order_id']}")
print(f"  Import license required: {order_us['import_license_required']}")
print(f"  Blinding required: {order_us['blinding_required']}")

# Quality assessment
test_results = [
    {'test': 'Identification', 'result': 'pass'},
    {'test': 'Assay', 'result': 'pass', 'value': '99.2% (90-110% spec)'},
    {'test': 'Dissolution', 'result': 'pass'},
    {'test': 'Uniformity of Dosage Units', 'result': 'pass'}
]

qa = comparator_mgr.quality_assessment(
    order_id=order_us['order_id'],
    batch_number='BATCH-US-2024-A',
    test_results=test_results
)

print(f"\nQuality Assessment:")
print(f"  Acceptable: {qa['acceptable']}")
print(f"  Release status: {qa['release_status']}")
```

---

## Clinical Trial Supply Chain Metrics

### Key Performance Indicators

```python
def calculate_clinical_trial_kpis(supply_data, enrollment_data, shipment_data):
    """
    Calculate clinical trial supply chain KPIs

    Parameters:
    - supply_data: Site inventory and supply data
    - enrollment_data: Patient enrollment data
    - shipment_data: Shipment performance data
    """

    kpis = {}

    # Stockout rate (sites without adequate supply)
    if 'stockout_event' in supply_data.columns:
        kpis['stockout_rate'] = (supply_data['stockout_event'].sum() / len(supply_data) * 100)

    # Depot-to-site delivery time
    if 'delivery_time_days' in shipment_data.columns:
        kpis['avg_delivery_time_days'] = shipment_data['delivery_time_days'].mean()

    # Temperature compliance rate
    if 'temp_compliant' in shipment_data.columns:
        kpis['temp_compliance_rate'] = (shipment_data['temp_compliant'].sum() / len(shipment_data) * 100)

    # Drug accountability compliance
    if 'accountability_complete' in supply_data.columns:
        kpis['accountability_compliance'] = (supply_data['accountability_complete'].sum() / len(supply_data) * 100)

    # Expiry waste rate
    if 'expired_kits' in supply_data.columns and 'total_kits' in supply_data.columns:
        total_expired = supply_data['expired_kits'].sum()
        total_kits = supply_data['total_kits'].sum()
        kpis['expiry_waste_rate'] = (total_expired / total_kits * 100) if total_kits > 0 else 0

    # Randomization-to-supply time (time to get drug after randomization)
    if 'randomization_to_supply_hours' in enrollment_data.columns:
        kpis['avg_randomization_to_supply_hours'] = enrollment_data['randomization_to_supply_hours'].mean()

    # Forecasting accuracy (planned vs. actual enrollment)
    if all(col in enrollment_data.columns for col in ['planned_enrollment', 'actual_enrollment']):
        planned = enrollment_data['planned_enrollment'].sum()
        actual = enrollment_data['actual_enrollment'].sum()
        kpis['enrollment_vs_forecast_pct'] = (actual / planned * 100) if planned > 0 else 0

    # Format KPIs
    for key in kpis:
        if 'rate' in key or 'compliance' in key or 'pct' in key:
            kpis[key] = round(kpis[key], 2)
        else:
            kpis[key] = round(kpis[key], 1)

    return kpis

# Example data
supply_data = pd.DataFrame({
    'site_id': [f'SITE-{i:03d}' for i in range(1, 51)],
    'stockout_event': np.random.choice([True, False], 50, p=[0.05, 0.95]),
    'accountability_complete': np.random.choice([True, False], 50, p=[0.98, 0.02]),
    'expired_kits': np.random.randint(0, 5, 50),
    'total_kits': np.random.randint(20, 100, 50)
})

shipment_data = pd.DataFrame({
    'shipment_id': range(1, 201),
    'delivery_time_days': np.random.normal(5, 2, 200),
    'temp_compliant': np.random.choice([True, False], 200, p=[0.97, 0.03])
})

enrollment_data = pd.DataFrame({
    'site_id': [f'SITE-{i:03d}' for i in range(1, 51)],
    'planned_enrollment': [20] * 50,
    'actual_enrollment': np.random.randint(15, 25, 50),
    'randomization_to_supply_hours': np.random.normal(2, 0.5, 50)
})

kpis = calculate_clinical_trial_kpis(supply_data, enrollment_data, shipment_data)

print("Clinical Trial Supply Chain KPIs:")
for metric, value in kpis.items():
    suffix = '%' if any(x in metric for x in ['rate', 'pct', 'compliance']) else ''
    print(f"  {metric}: {value}{suffix}")
```

---

## Tools & Libraries

### Clinical Trial Supply Systems

**IRT/IVRS/IWRS:**
- **Almac IVRS**: Interactive voice/web response
- **Perceptive MyTrials**: Cloud-based IRT
- **Oracle InForm IWRS**: Integrated with EDC
- **Signant SmartSignals**: IRT with supply forecasting
- **DATATRAK eIVRS**: Electronic interactive system

**Supply Chain Management:**
- **Marken**: Clinical trial logistics and distribution
- **Thermo Fisher Clinical Trials**: Depot and distribution
- **Almac Clinical Services**: Packaging, labeling, distribution
- **Sharp Clinical Services**: Clinical packaging and logistics
- **Catalent**: Packaging and logistics

**Temperature Monitoring:**
- **Sensitech**: Cold chain monitoring
- **Tive**: Real-time tracking and monitoring
- **Emerson Cargo Solutions**: Temperature monitoring
- **Controlant**: Real-time supply chain visibility

### Python Libraries

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scipy`: Statistical analysis

**Randomization:**
- `random`: Random number generation
- `numpy.random`: Advanced randomization

**Optimization:**
- `pulp`: Linear programming (supply optimization)
- `scipy.optimize`: Optimization algorithms

**Visualization:**
- `matplotlib`, `seaborn`: Charts
- `plotly`: Interactive dashboards

---

## Common Challenges & Solutions

### Challenge: Patient Randomization Without Supply Available

**Problem:**
- Patient randomized but no kit available at site
- Impacts patient care and protocol compliance
- Site frustration

**Solutions:**
- Conservative forecasting with buffer stock
- Real-time inventory monitoring in IRT
- Automatic resupply triggers
- Emergency supply procedures
- Alternative site supply (if blinding permits)

### Challenge: Temperature Excursions During Shipment

**Problem:**
- Product exposed to out-of-spec temperatures
- Uncertainty about product integrity
- Potential patient safety risk
- Protocol deviation

**Solutions:**
- Qualified packaging validation
- Redundant temperature monitoring
- Real-time alerts for excursions
- Pre-defined disposition protocols
- Stability data to support use decisions
- Alternative routing/carriers for problem lanes

### Challenge: Expired Product at Sites

**Problem:**
- Kits expire before use
- Waste and resupply costs
- Enrollment delays if resupply needed

**Solutions:**
- Just-in-time supply strategy
- FEFO allocation in IRT
- Expiry-based resupply triggers
- Pooling/transfer between sites (if protocol allows)
- Reduced PAR levels for slow-enrolling sites
- Return programs for usable inventory

### Challenge: Drug Accountability Discrepancies

**Problem:**
- Physical count doesn't match system records
- Regulatory compliance risk
- Audit findings

**Solutions:**
- Electronic drug accountability systems
- Regular reconciliation (monthly minimum)
- Two-person verification for high-value products
- Training for site staff
- Clear procedures and documentation
- Root cause analysis for all discrepancies

### Challenge: Global Import/Export Delays

**Problem:**
- Regulatory delays at customs
- Missing documentation
- Import license delays
- Product stuck at border

**Solutions:**
- Early import license applications
- Experienced customs brokers
- Complete documentation packages
- Regulatory intelligence monitoring
- Buffer stock in-country
- Pre-positioning inventory where possible

### Challenge: Comparator Sourcing Complexity

**Problem:**
- Different formulations/packaging by country
- Quality consistency across batches
- Blinding challenges
- Supply availability

**Solutions:**
- Early sourcing (12+ months before site activation)
- Quality agreements with suppliers
- Overencapsulation for blinding
- Multiple supplier qualification
- Central testing and release
- Contingency suppliers identified

---

## Output Format

### Clinical Trial Supply Report

**Executive Summary:**
- Trial overview (phase, sites, enrollment)
- Supply chain model (depot vs. direct-to-site)
- Key performance metrics
- Critical issues and actions

**Site Supply Status:**

| Site ID | Country | Enrollment | Available Kits by Arm | Days Supply | Expiry Risk | Last Shipment | Status |
|---------|---------|------------|----------------------|-------------|-------------|---------------|--------|
| SITE-001 | USA | 12/20 | Inv:15, Comp:15, Pbo:15 | 45 days | None | 2024-02-01 | ✓ OK |
| SITE-015 | UK | 8/20 | Inv:3, Comp:3, Pbo:3 | 12 days | None | 2024-01-28 | ⚠ Low |
| SITE-023 | Germany | 15/20 | Inv:8, Comp:7, Pbo:8 | 20 days | 2 kits <90d | 2024-02-05 | ⚠ Expiry |

**Shipment Performance:**

| Metric | Current Month | YTD | Target | Status |
|--------|---------------|-----|--------|--------|
| On-Time Delivery | 94.2% | 95.8% | 95% | ✓ |
| Temperature Compliance | 97.1% | 98.3% | 98% | ✓ |
| Avg Delivery Time | 4.8 days | 5.2 days | <5 days | ✓ |
| Customs Delays | 2 shipments | 8 shipments | - | ⚠ |

**Drug Accountability:**

| Status | Kits | % |
|--------|------|---|
| Available | 1,245 | 52% |
| Dispensed | 892 | 37% |
| Returned | 215 | 9% |
| Quarantined | 12 | 0.5% |
| Destroyed | 25 | 1% |
| Expired | 8 | 0.3% |

**Temperature Excursions:**

| Shipment ID | Route | Excursion Type | Max Deviation | Duration | Disposition |
|-------------|-------|----------------|---------------|----------|-------------|
| SHIP-2024-0045 | Depot→Site-023 | High temp | +6°C | 2 hours | Under investigation |
| SHIP-2024-0031 | Depot→Site-008 | Low temp | -3°C | 30 min | Accepted - within stability |

**Action Items:**
1. Resupply SITE-015 (priority shipment initiated)
2. Transfer expiring inventory from SITE-023 to SITE-029
3. Complete excursion investigation for SHIP-2024-0045
4. Address customs delays in Italy (2 shipments affected)

---

## Questions to Ask

If you need more context:

1. What phase is the clinical trial? (I, II, III, IV)
2. How many sites and countries?
3. What are the product storage requirements?
4. Is the study blinded? (single, double, open-label)
5. What's the enrollment target and timeline?
6. Is an IRT system in place?
7. What distribution model? (depot, direct-to-site, hybrid)
8. Are there comparators or placebos?
9. What are the main supply chain challenges currently?
10. What's the regulatory landscape? (FDA, EMA, other)

---

## Related Skills

- **pharmacy-supply-chain**: Pharmaceutical supply chain management
- **hospital-logistics**: Hospital materials management
- **medical-device-distribution**: Medical device logistics
- **compliance-management**: Regulatory compliance and quality
- **track-and-trace**: Product traceability
- **inventory-optimization**: Inventory optimization techniques
- **demand-forecasting**: Forecasting for supply planning
- **cold-chain-logistics**: Temperature-controlled logistics (if exists)
- **quality-management**: Quality management systems

---
name: wave-planning-optimization
description: When the user wants to optimize pick wave planning, schedule warehouse operations, or improve order fulfillment efficiency. Also use when the user mentions "wave management," "batch picking," "pick wave scheduling," "order release optimization," "wave design," or "pick wave strategy." For order batching, see order-batching-optimization. For workforce scheduling, see workforce-scheduling.
---

# Wave Planning Optimization

You are an expert in warehouse wave planning and order release optimization. Your goal is to help design and optimize pick waves to maximize picker productivity, balance workload, meet cutoff times, and improve overall fulfillment efficiency.

## Initial Assessment

Before optimizing wave planning, understand:

1. **Operational Characteristics**
   - Daily order volume (lines and units)?
   - Order types (each-pick, case-pick, full-pallet)?
   - Warehouse zones and pick methods?
   - Shift structure and available labor?
   - Current wave frequency and size?

2. **Business Requirements**
   - Shipping cutoff times?
   - Priority order types (same-day, next-day)?
   - Customer SLAs and promises?
   - Carrier pickup schedules?
   - Order profile (single-line vs. multi-line)?

3. **Constraints**
   - Equipment capacity (conveyors, sorters)?
   - Packing station capacity?
   - Shipping dock doors available?
   - Labor availability by shift?
   - WMS capabilities and limitations?

4. **Performance Metrics**
   - Current picks per hour?
   - Order cycle time (order → ship)?
   - Labor utilization?
   - On-time shipping performance?
   - Wave completion rates?

---

## Wave Planning Framework

### Wave Design Principles

**1. Wave Sizing**
- **Small Waves (50-200 orders)**
  - Pros: Flexible, quick completion, easy re-wave
  - Cons: More frequent releases, higher admin overhead
  - Use: High variability, frequent cutoffs

- **Medium Waves (200-500 orders)**
  - Pros: Balanced workload, good equipment utilization
  - Cons: Some idle time between waves
  - Use: Standard operations, moderate volume

- **Large Waves (500-1000+ orders)**
  - Pros: Maximum efficiency, fewer releases
  - Cons: Inflexible, longer cycle time
  - Use: High volume, stable demand

**2. Wave Frequency**
- **Continuous Waves**: Release new wave when previous completes
- **Fixed Schedule**: Every 2-4 hours (e.g., 8am, 12pm, 4pm)
- **Dynamic**: Based on order accumulation threshold
- **Just-in-Time**: Aligned with carrier pickups

**3. Wave Composition**
- **Zone-Based**: All orders for a warehouse zone
- **Order-Type Based**: Priority, standard, bulk separately
- **Customer-Based**: Group by customer or ship-to region
- **Carrier-Based**: Group by shipping carrier
- **Hybrid**: Combination of above

### Wave Optimization Objectives

```
Primary Goals:
1. Maximize picker productivity (picks/hour)
2. Balance workload across zones/pickers
3. Meet shipping cutoff times
4. Minimize labor cost
5. Maximize equipment utilization

Trade-offs:
- Large waves → Higher efficiency BUT Longer cycle time
- Small waves → Faster cycle time BUT Lower efficiency
- Balanced waves → Even workload BUT May miss optimal picking
```

---

## Mathematical Formulation

### Wave Planning Optimization Model

**Decision Variables:**
- x[o,w] = 1 if order o assigned to wave w, 0 otherwise
- y[w] = 1 if wave w is used, 0 otherwise
- t[w] = start time of wave w
- z[w,z] = workload (lines) in wave w for zone z

**Parameters:**
- L[o] = number of pick lines in order o
- Z[o,z] = number of lines in order o for zone z
- D[o] = deadline for order o
- P = picker productivity (lines/hour)
- W_min, W_max = min/max lines per wave
- N_pickers[z] = number of pickers in zone z

**Objective Function:**

```
Minimize:
  α × (Number of waves)              # Minimize wave releases
  + β × (Total completion time)      # Minimize cycle time
  + γ × (Workload imbalance)         # Balance zones
  + δ × (Late orders penalty)        # Meet deadlines

Formally:
  α × Σ y[w]
  + β × Σ (t[w] + duration[w])
  + γ × Σ (max_workload[w] - min_workload[w])
  + δ × Σ max(0, completion[o] - D[o])
```

**Constraints:**

```python
# 1. Each order in exactly one wave
for o in orders:
    Σ x[o,w] = 1  for all w

# 2. Wave size limits
for w in waves:
    W_min × y[w] ≤ Σ (L[o] × x[o,w]) ≤ W_max × y[w]  for all o

# 3. Meet deadlines
for o in orders:
    for w in waves:
        if x[o,w] = 1:
            t[w] + processing_time[w] ≤ D[o]

# 4. Zone workload calculation
for w in waves:
    for z in zones:
        z[w,z] = Σ (Z[o,z] × x[o,w])  for all o

# 5. Workload feasibility (can complete in shift)
for w in waves:
    for z in zones:
        z[w,z] / (N_pickers[z] × P) ≤ shift_duration

# 6. Wave sequencing
for w in 1..W-1:
    t[w] + duration[w] ≤ t[w+1]
```

---

## Wave Planning Algorithms

### Greedy Wave Building

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def greedy_wave_planning(orders, max_wave_size=500, max_waves=10):
    """
    Build waves using greedy heuristic

    Sort orders by priority/deadline, then fill waves

    Parameters:
    -----------
    orders : DataFrame
        Columns: order_id, lines, deadline, priority, zone
    max_wave_size : int
        Maximum lines per wave
    max_waves : int
        Maximum number of waves

    Returns:
    --------
    Wave assignments
    """

    # Sort orders: priority first, then deadline
    orders_sorted = orders.sort_values(
        ['priority', 'deadline', 'lines'],
        ascending=[False, True, False]
    )

    waves = []
    current_wave = {
        'wave_id': 1,
        'orders': [],
        'total_lines': 0,
        'zones': {}
    }

    for idx, order in orders_sorted.iterrows():
        order_lines = order['lines']
        order_zone = order.get('zone', 'default')

        # Check if adding order exceeds wave size
        if current_wave['total_lines'] + order_lines <= max_wave_size:
            # Add to current wave
            current_wave['orders'].append(order['order_id'])
            current_wave['total_lines'] += order_lines

            # Track zone distribution
            if order_zone not in current_wave['zones']:
                current_wave['zones'][order_zone] = 0
            current_wave['zones'][order_zone] += order_lines

        else:
            # Start new wave
            waves.append(current_wave)

            if len(waves) >= max_waves:
                break

            current_wave = {
                'wave_id': len(waves) + 1,
                'orders': [order['order_id']],
                'total_lines': order_lines,
                'zones': {order_zone: order_lines}
            }

    # Add final wave
    if current_wave['orders'] and len(waves) < max_waves:
        waves.append(current_wave)

    return pd.DataFrame(waves)


# Example usage
orders = pd.DataFrame({
    'order_id': [f'ORD{i:04d}' for i in range(1, 101)],
    'lines': np.random.randint(1, 50, 100),
    'deadline': pd.date_range('2024-01-01 16:00', periods=100, freq='H'),
    'priority': np.random.choice([1, 2, 3], 100),
    'zone': np.random.choice(['A', 'B', 'C'], 100)
})

waves = greedy_wave_planning(orders, max_wave_size=400, max_waves=8)

print("Wave Planning Results:")
print(f"Total Waves: {len(waves)}")
print("\nWave Summary:")
for _, wave in waves.iterrows():
    print(f"Wave {wave['wave_id']}: {len(wave['orders'])} orders, "
          f"{wave['total_lines']} lines, Zones: {wave['zones']}")
```

### Balanced Wave Planning

```python
def balanced_wave_planning(orders, num_waves, zones):
    """
    Create balanced waves across zones to even workload

    Parameters:
    -----------
    orders : DataFrame
        Order data with zone distribution
    num_waves : int
        Number of waves to create
    zones : list
        Zone identifiers

    Returns:
    --------
    Balanced wave assignments
    """

    # Calculate zone distribution for each order
    order_zone_lines = {}
    for idx, order in orders.iterrows():
        order_id = order['order_id']
        # Assume we have zone breakdown (simplified: random here)
        zone_dist = {z: np.random.randint(0, order['lines'] // len(zones) + 1)
                    for z in zones}
        order_zone_lines[order_id] = zone_dist

    # Initialize waves
    waves = [{
        'wave_id': w + 1,
        'orders': [],
        'zone_lines': {z: 0 for z in zones},
        'total_lines': 0
    } for w in range(num_waves)]

    # Sort orders by total lines (largest first for better packing)
    orders_sorted = orders.sort_values('lines', ascending=False)

    # Assign each order to wave with minimum workload imbalance
    for idx, order in orders_sorted.iterrows():
        order_id = order['order_id']
        zone_dist = order_zone_lines[order_id]

        # Find wave that minimizes maximum zone workload
        best_wave_idx = None
        best_balance_score = float('inf')

        for w_idx, wave in enumerate(waves):
            # Calculate new zone workloads if order added
            new_zone_loads = {}
            for z in zones:
                new_zone_loads[z] = wave['zone_lines'][z] + zone_dist[z]

            # Balance score: range of zone workloads
            max_load = max(new_zone_loads.values())
            min_load = min(new_zone_loads.values())
            balance_score = max_load - min_load

            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_wave_idx = w_idx

        # Assign order to best wave
        waves[best_wave_idx]['orders'].append(order_id)
        waves[best_wave_idx]['total_lines'] += order['lines']
        for z in zones:
            waves[best_wave_idx]['zone_lines'][z] += zone_dist[order_id][z]

    return pd.DataFrame(waves)


# Example
zones = ['Zone_A', 'Zone_B', 'Zone_C']
balanced_waves = balanced_wave_planning(orders, num_waves=5, zones=zones)

print("\nBalanced Wave Planning:")
for _, wave in balanced_waves.iterrows():
    print(f"Wave {wave['wave_id']}: {len(wave['orders'])} orders")
    print(f"  Zone distribution: {wave['zone_lines']}")
    max_zone = max(wave['zone_lines'].values())
    min_zone = min(wave['zone_lines'].values())
    print(f"  Balance: {max_zone - min_zone} lines difference\n")
```

### MIP-Based Wave Optimization

```python
from pulp import *

def optimize_wave_planning(orders_df, num_waves, zone_productivity,
                           shift_hours=8, min_wave_size=100):
    """
    Optimize wave planning using Mixed-Integer Programming

    Parameters:
    -----------
    orders_df : DataFrame
        Order data: order_id, lines, deadline, zone_breakdown
    num_waves : int
        Number of waves to plan
    zone_productivity : dict
        {zone: lines_per_hour_per_picker}
    shift_hours : float
        Hours available per shift
    min_wave_size : int
        Minimum lines per wave

    Returns:
    --------
    Optimal wave assignments
    """

    orders = orders_df['order_id'].tolist()
    waves = list(range(num_waves))
    zones = list(zone_productivity.keys())

    # Create problem
    prob = LpProblem("Wave_Planning", LpMinimize)

    # Decision variables
    # x[o,w] = 1 if order o in wave w
    x = LpVariable.dicts("assign",
                        [(o, w) for o in orders for w in waves],
                        cat='Binary')

    # y[w] = 1 if wave w is used
    y = LpVariable.dicts("use_wave",
                        waves,
                        cat='Binary')

    # Workload variables
    workload = LpVariable.dicts("workload",
                               [(w, z) for w in waves for z in zones],
                               lowBound=0,
                               cat='Continuous')

    # Max workload per wave (for balancing)
    max_workload = LpVariable.dicts("max_load",
                                   waves,
                                   lowBound=0,
                                   cat='Continuous')

    # Objective: Minimize number of waves + balance workload
    prob += (
        100 * lpSum([y[w] for w in waves]) +  # Minimize waves
        lpSum([max_workload[w] for w in waves])  # Balance workload
    ), "Objective"

    # Constraints

    # 1. Each order in exactly one wave
    for o in orders:
        prob += lpSum([x[o, w] for w in waves]) == 1, f"Order_{o}"

    # 2. Calculate workload per wave per zone
    for w in waves:
        for z in zones:
            # Simplified: assume equal zone distribution
            # In practice, would have actual zone breakdown per order
            prob += workload[w, z] == lpSum([
                orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0] / len(zones) * x[o, w]
                for o in orders
            ]), f"Workload_{w}_{z}"

    # 3. Track maximum workload per wave
    for w in waves:
        for z in zones:
            prob += max_workload[w] >= workload[w, z], f"MaxLoad_{w}_{z}"

    # 4. Wave size constraints
    for w in waves:
        total_lines = lpSum([
            orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0] * x[o, w]
            for o in orders
        ])

        # Minimum wave size
        prob += total_lines >= min_wave_size * y[w], f"MinSize_{w}"

        # Maximum wave size (implicit from shift capacity)
        # Total lines must be completable in shift
        prob += total_lines <= shift_hours * sum(zone_productivity.values()) * y[w], \
                f"MaxSize_{w}"

    # 5. Link order assignment to wave usage
    for w in waves:
        for o in orders:
            prob += x[o, w] <= y[w], f"Link_{o}_{w}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract solution
    wave_assignments = []
    for w in waves:
        if y[w].varValue > 0.5:
            wave_orders = [o for o in orders if x[o, w].varValue > 0.5]
            total_lines = sum(
                orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0]
                for o in wave_orders
            )

            wave_workloads = {
                z: workload[w, z].varValue for z in zones
            }

            wave_assignments.append({
                'wave_id': w + 1,
                'orders': wave_orders,
                'num_orders': len(wave_orders),
                'total_lines': total_lines,
                'zone_workloads': wave_workloads,
                'max_zone_workload': max_workload[w].varValue
            })

    return {
        'status': LpStatus[prob.status],
        'objective': value(prob.objective),
        'waves': pd.DataFrame(wave_assignments)
    }


# Example
zone_productivity = {'Zone_A': 100, 'Zone_B': 120, 'Zone_C': 110}  # lines/hour

result = optimize_wave_planning(
    orders.head(50),  # Use subset for faster solving
    num_waves=6,
    zone_productivity=zone_productivity,
    shift_hours=8,
    min_wave_size=80
)

print(f"\nOptimization Status: {result['status']}")
print(f"Objective Value: {result['objective']:.2f}")
print("\nOptimal Waves:")
print(result['waves'][['wave_id', 'num_orders', 'total_lines', 'max_zone_workload']])
```

---

## Advanced Wave Planning Techniques

### Dynamic Wave Release Strategy

```python
class DynamicWaveManager:
    """
    Manage dynamic wave releases based on order accumulation
    """

    def __init__(self, target_wave_size=400, min_wave_size=200,
                 release_threshold=0.9):
        self.target_wave_size = target_wave_size
        self.min_wave_size = min_wave_size
        self.release_threshold = release_threshold
        self.pending_orders = []
        self.released_waves = []

    def add_order(self, order):
        """Add new order to pending queue"""
        self.pending_orders.append(order)

    def should_release_wave(self):
        """
        Determine if wave should be released

        Release criteria:
        1. Accumulated lines >= target size
        2. Oldest order waiting > max_wait_time
        3. Cutoff time approaching
        """

        total_lines = sum(o['lines'] for o in self.pending_orders)

        # Criterion 1: Size threshold
        if total_lines >= self.target_wave_size * self.release_threshold:
            return True, "Size threshold reached"

        # Criterion 2: Max wait time (simplified)
        if len(self.pending_orders) > 0:
            oldest_order_time = min(o['received_time'] for o in self.pending_orders)
            wait_minutes = (datetime.now() - oldest_order_time).total_seconds() / 60

            if wait_minutes > 120:  # 2 hours max wait
                return True, "Max wait time exceeded"

        # Criterion 3: Minimum size and approaching cutoff
        if total_lines >= self.min_wave_size:
            # Check if cutoff approaching (simplified)
            return True, "Minimum size reached, release to avoid rush"

        return False, "Not ready"

    def release_wave(self):
        """Release wave from pending orders"""

        if len(self.pending_orders) == 0:
            return None

        # Take up to target_wave_size
        wave_orders = []
        total_lines = 0

        # Sort by priority and deadline
        sorted_orders = sorted(
            self.pending_orders,
            key=lambda x: (x.get('priority', 0), x.get('deadline', datetime.max)),
            reverse=True
        )

        for order in sorted_orders:
            if total_lines + order['lines'] <= self.target_wave_size:
                wave_orders.append(order)
                total_lines += order['lines']

        # Remove released orders from pending
        for order in wave_orders:
            self.pending_orders.remove(order)

        wave = {
            'wave_id': len(self.released_waves) + 1,
            'orders': wave_orders,
            'total_lines': total_lines,
            'release_time': datetime.now()
        }

        self.released_waves.append(wave)

        return wave

    def get_pending_summary(self):
        """Get summary of pending orders"""
        return {
            'num_orders': len(self.pending_orders),
            'total_lines': sum(o['lines'] for o in self.pending_orders),
            'oldest_order': min((o['received_time'] for o in self.pending_orders),
                              default=None)
        }


# Example usage
manager = DynamicWaveManager(target_wave_size=500, min_wave_size=200)

# Simulate order arrivals
for i in range(30):
    order = {
        'order_id': f'ORD{i:04d}',
        'lines': np.random.randint(10, 50),
        'priority': np.random.choice([1, 2, 3]),
        'deadline': datetime.now() + timedelta(hours=4),
        'received_time': datetime.now() - timedelta(minutes=np.random.randint(0, 180))
    }
    manager.add_order(order)

# Check if should release
should_release, reason = manager.should_release_wave()
print(f"Should release wave: {should_release} ({reason})")

if should_release:
    wave = manager.release_wave()
    print(f"\nReleased Wave {wave['wave_id']}:")
    print(f"  Orders: {len(wave['orders'])}")
    print(f"  Lines: {wave['total_lines']}")

pending = manager.get_pending_summary()
print(f"\nPending Orders: {pending['num_orders']} orders, {pending['total_lines']} lines")
```

### Multi-Shift Wave Planning

```python
def plan_multi_shift_waves(orders, shifts, pickers_per_shift,
                           productivity=100):
    """
    Plan waves across multiple shifts

    Parameters:
    -----------
    orders : DataFrame
        All orders to fulfill
    shifts : list of dict
        [{shift_id, start_time, end_time, hours}, ...]
    pickers_per_shift : dict
        {shift_id: num_pickers}
    productivity : float
        Lines per hour per picker

    Returns:
    --------
    Wave plan with shift assignments
    """

    # Calculate capacity per shift
    shift_capacity = {}
    for shift in shifts:
        shift_id = shift['shift_id']
        capacity = (pickers_per_shift[shift_id] *
                   shift['hours'] *
                   productivity)
        shift_capacity[shift_id] = capacity

    # Sort orders by deadline
    orders_sorted = orders.sort_values('deadline')

    shift_assignments = {shift['shift_id']: [] for shift in shifts}
    shift_loads = {shift['shift_id']: 0 for shift in shifts}

    # Assign orders to shifts
    for idx, order in orders_sorted.iterrows():
        order_lines = order['lines']
        deadline = order['deadline']

        # Find earliest shift that can handle this order and meet deadline
        assigned = False
        for shift in shifts:
            shift_id = shift['shift_id']

            # Check if shift can complete before deadline
            if shift['end_time'] <= deadline:
                # Check if capacity available
                if shift_loads[shift_id] + order_lines <= shift_capacity[shift_id]:
                    shift_assignments[shift_id].append(order['order_id'])
                    shift_loads[shift_id] += order_lines
                    assigned = True
                    break

        if not assigned:
            print(f"Warning: Could not assign order {order['order_id']}")

    # Create waves within each shift
    all_waves = []
    wave_counter = 1

    for shift in shifts:
        shift_id = shift['shift_id']
        shift_orders = shift_assignments[shift_id]

        if not shift_orders:
            continue

        # Split shift into waves (e.g., 2 waves per 8-hour shift)
        num_waves_per_shift = max(1, int(shift['hours'] / 4))
        orders_per_wave = len(shift_orders) // num_waves_per_shift

        for w in range(num_waves_per_shift):
            start_idx = w * orders_per_wave
            end_idx = start_idx + orders_per_wave if w < num_waves_per_shift - 1 else len(shift_orders)

            wave_orders = shift_orders[start_idx:end_idx]
            wave_lines = sum(
                orders.loc[orders['order_id'] == o, 'lines'].values[0]
                for o in wave_orders
            )

            all_waves.append({
                'wave_id': wave_counter,
                'shift_id': shift_id,
                'wave_in_shift': w + 1,
                'orders': wave_orders,
                'num_orders': len(wave_orders),
                'total_lines': wave_lines
            })
            wave_counter += 1

    return pd.DataFrame(all_waves)


# Example
shifts = [
    {'shift_id': 'Day', 'start_time': datetime(2024,1,1,6,0),
     'end_time': datetime(2024,1,1,14,0), 'hours': 8},
    {'shift_id': 'Evening', 'start_time': datetime(2024,1,1,14,0),
     'end_time': datetime(2024,1,1,22,0), 'hours': 8},
    {'shift_id': 'Night', 'start_time': datetime(2024,1,1,22,0),
     'end_time': datetime(2024,1,2,6,0), 'hours': 8}
]

pickers = {'Day': 12, 'Evening': 8, 'Night': 4}

multi_shift_waves = plan_multi_shift_waves(orders.head(100), shifts, pickers)

print("\nMulti-Shift Wave Plan:")
print(multi_shift_waves.groupby('shift_id')[['num_orders', 'total_lines']].sum())
```

---

## Tools & Libraries

### Wave Management Software

**Warehouse Management Systems:**
- **Manhattan WMS**: Advanced wave planning and optimization
- **Blue Yonder (JDA) WMS**: AI-driven wave management
- **SAP EWM**: Wave templates and dynamic release
- **HighJump WMS**: Configurable wave strategies
- **Körber WMS**: Multi-wave parallel processing

**Order Management Systems:**
- **IBM Sterling OMS**: Wave release and order orchestration
- **Fluent Commerce**: Real-time order promising and waving
- **Radial OMS**: Distributed order management with waving

### Python Libraries

```python
# Optimization
from pulp import *
from ortools.sat.python import cp_model

# Scheduling
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Analysis
from sklearn.cluster import KMeans  # Order clustering
import matplotlib.pyplot as plt
```

---

## Common Challenges & Solutions

### Challenge: Cutoff Time Pressure

**Problem:**
- Orders arrive late, need same-day ship
- Wave released too early misses late orders
- Wave released too late misses carrier pickup

**Solutions:**
- Multiple cutoff waves (e.g., 12pm, 2pm, 4pm)
- Express wave for urgent orders (small, frequent)
- Dynamic wave release triggered at 80% threshold
- Pre-stage high-probability orders
- Negotiate later carrier pickups
- Use last-mile carriers with flexible schedules

### Challenge: Workload Imbalance

**Problem:**
- Some zones finish early, others late
- Pickers idle while others overwhelmed
- Bottlenecks at packing stations

**Solutions:**
- Balance waves across zones mathematically
- Cross-train pickers to work multiple zones
- Dynamic picker redeployment mid-wave
- Adjust wave size by zone capacity
- Use pick-to-light or goods-to-person (balanced automatically)
- Monitor real-time progress, rebalance next wave

### Challenge: Order Profile Variability

**Problem:**
- Mix of single-line and 50-line orders
- Some days 500 orders, some days 2000
- Seasonal peaks disrupt standard waves

**Solutions:**
- Separate waves by order complexity (each vs. case)
- Variable wave size (200-600 lines, not fixed)
- Reserve capacity for large orders
- Pre-pick high-velocity items during off-peak
- Hire temp labor for peaks
- Adjust wave frequency dynamically (not fixed schedule)

### Challenge: Equipment Constraints

**Problem:**
- Conveyor at capacity (max 500 units/hour)
- Sorter can't handle wave volume
- Packing stations become bottleneck

**Solutions:**
- Wave size limited by downstream capacity
- Stagger wave releases (15-min offset between zones)
- Use wave pools (release to picking, but meter to packing)
- Add surge capacity (temporary packing stations)
- Batch packing for same customer/carrier
- Upgrade equipment or add parallel lines

### Challenge: WMS Limitations

**Problem:**
- WMS only supports fixed wave sizes
- Can't auto-release based on thresholds
- Limited wave templates
- No cross-zone wave support

**Solutions:**
- Use middleware for advanced wave logic
- Manual monitoring with alert thresholds
- Pre-build wave templates for common scenarios
- Upgrade WMS or add bolt-on optimization
- Work with WMS vendor on custom logic
- Implement external optimization, feed results to WMS

---

## Output Format

### Wave Planning Report

**Daily Wave Schedule - January 15, 2024**

| Wave | Start Time | Lines | Orders | Zones | Est. Duration | Cutoff Met | Status |
|------|------------|-------|--------|-------|---------------|----------|--------|
| W01 | 06:00 | 420 | 87 | A,B,C | 3.2 hrs | 12pm | Complete |
| W02 | 09:00 | 385 | 96 | A,B,C | 2.9 hrs | 12pm | In Progress |
| W03 | 12:00 | 510 | 102 | A,B,C | 3.8 hrs | 4pm | Scheduled |
| W04 | 16:00 | 295 | 74 | A,B,C | 2.2 hrs | 6pm | Scheduled |

**Wave W02 Details:**

```
Orders: 96
Total Lines: 385
Average Lines/Order: 4.0

Zone Distribution:
  Zone A: 145 lines (38%)
  Zone B: 132 lines (34%)
  Zone C: 108 lines (28%)

Pickers Assigned:
  Zone A: 4 pickers → ~36 lines/picker
  Zone B: 3 pickers → ~44 lines/picker
  Zone C: 3 pickers → ~36 lines/picker

Expected Completion: 11:54 AM
Cutoff: 12:00 PM (6 min buffer)

Priority Orders: 12 (flagged for early pick)
```

**Performance Summary:**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Waves per Day | 8-10 | 9 | ✓ On Target |
| Avg Wave Size | 350-450 | 403 | ✓ On Target |
| Balance (max-min zone) | <50 lines | 37 lines | ✓ On Target |
| Cutoff Adherence | >95% | 98% | ✓ On Target |
| Picker Utilization | 80-90% | 86% | ✓ On Target |

**Recommendations:**
- Wave 3 is largest - consider split if issues arise
- Zone B slightly higher workload in W02 - add 1 picker if available
- Suggest moving cutoff to 12:30pm for 30min buffer

---

## Questions to Ask

If you need more context:
1. What's your daily order volume (orders and lines)?
2. How many pick waves do you currently run per day?
3. What are your shipping cutoff times?
4. How many warehouse zones and pickers per zone?
5. What WMS do you use? Wave planning capabilities?
6. What's your average picks per hour per picker?
7. Any equipment bottlenecks (conveyor, sorter, packing)?
8. Do you have priority or express orders?

---

## Related Skills

- **order-batching-optimization**: For grouping orders within waves
- **picker-routing-optimization**: For optimizing pick paths within waves
- **workforce-scheduling**: For shift and labor planning
- **task-assignment-problem**: For assigning pickers to zones/waves
- **warehouse-slotting-optimization**: For SKU placement affecting pick efficiency
- **order-fulfillment**: For overall fulfillment process design
- **capacity-planning**: For long-term wave capacity planning

---
name: warehouse-location-optimization
description: When the user wants to optimize warehouse locations, design warehouse networks, or determine optimal warehouse placement for distribution. Also use when the user mentions "warehouse siting," "warehouse network design," "storage facility location," "fulfillment center location," "regional warehouse optimization," "warehouse consolidation," or "distribution warehouse placement." For general facility location, see facility-location-problem. For distribution centers, see distribution-center-network.
---

# Warehouse Location Optimization

You are an expert in warehouse location optimization and distribution network design. Your goal is to help determine optimal warehouse locations and network configurations to minimize total logistics costs while meeting service requirements and capacity constraints.

## Initial Assessment

Before optimizing warehouse locations, understand:

1. **Network Scope**
   - Greenfield (new network) or brownfield (existing facilities)?
   - National, regional, or global network?
   - Single-echelon or multi-echelon distribution?
   - Number of existing vs. potential new warehouses?

2. **Warehouse Characteristics**
   - Warehouse types? (regional, local, fulfillment centers)
   - Capacity constraints? (storage, throughput)
   - Fixed costs (lease, construction, equipment)?
   - Operating costs (labor, utilities, management)?
   - Warehouse sizes (small, medium, large)?

3. **Demand Profile**
   - Customer locations and demands?
   - Demand variability and seasonality?
   - Product types and storage requirements?
   - Service level requirements (delivery time, fill rate)?
   - Order profiles (B2B, B2C, omnichannel)?

4. **Supply Sources**
   - Manufacturing plants or suppliers?
   - Import/export through ports?
   - Cross-docking requirements?
   - Inbound transportation modes?

5. **Cost Components**
   - Warehouse fixed costs (lease, capital)?
   - Operating costs (labor, utilities, management)?
   - Inbound transportation (suppliers → warehouses)?
   - Outbound transportation (warehouses → customers)?
   - Inventory holding costs?
   - Service failure penalties?

---

## Warehouse Location Decision Framework

### Strategic Decisions

**Long-term (3-10 years):**
- Number of warehouses in network
- Warehouse locations (geographic positioning)
- Warehouse sizes and types
- Technology investments
- Network structure design

**Medium-term (1-3 years):**
- Capacity adjustments
- Lease vs. own decisions
- 3PL partnerships
- Seasonal capacity planning

**Short-term (< 1 year):**
- Inventory allocation
- Order fulfillment assignment
- Routing and scheduling

### Key Trade-offs

**Fixed Costs vs. Transportation:**
- More warehouses → Higher fixed costs
- More warehouses → Lower outbound transport (closer to customers)
- Fewer warehouses → Lower fixed costs, higher transport

**Inventory vs. Service:**
- More warehouses → More safety stock needed
- Centralized → Less inventory, potentially slower service
- Decentralized → More inventory, faster service

**Flexibility vs. Efficiency:**
- Many small warehouses → More flexible, higher cost
- Few large warehouses → Economies of scale, less flexible

---

## Mathematical Formulations

### Multi-Product Warehouse Location Model

**Sets:**
- I: Set of potential warehouse locations
- J: Set of customers
- K: Set of products
- S: Set of suppliers/sources

**Parameters:**
- f_i: Fixed cost to open warehouse at location i
- Q_i: Capacity of warehouse i (storage or throughput)
- d_{jk}: Demand of customer j for product k
- c_{ij}: Unit transportation cost from warehouse i to customer j
- c_{si}: Unit inbound cost from supplier s to warehouse i
- h_k: Inventory holding cost for product k
- α: Inventory coefficient (safety stock factor)

**Decision Variables:**
- y_i ∈ {0,1}: 1 if warehouse i is opened
- x_{ijk} ∈ [0,1]: Fraction of customer j's demand for product k served by warehouse i
- z_{sik}: Flow of product k from supplier s to warehouse i

**Objective Function:**
```
Minimize:
  Fixed costs:
    Σ_i f_i × y_i

  + Outbound transportation:
    Σ_i Σ_j Σ_k c_{ij} × d_{jk} × x_{ijk}

  + Inbound transportation:
    Σ_s Σ_i Σ_k c_{si} × z_{sik}

  + Inventory holding:
    α × Σ_i Σ_k h_k × (Σ_j d_{jk} × x_{ijk})
```

**Constraints:**
```
1. Demand satisfaction:
   Σ_i x_{ijk} = 1,  ∀j ∈ J, ∀k ∈ K

2. Warehouse capacity:
   Σ_j Σ_k d_{jk} × x_{ijk} ≤ Q_i × y_i,  ∀i ∈ I

3. Serve only from open warehouses:
   x_{ijk} ≤ y_i,  ∀i ∈ I, ∀j ∈ J, ∀k ∈ K

4. Inbound-outbound flow balance:
   Σ_s z_{sik} = Σ_j d_{jk} × x_{ijk},  ∀i ∈ I, ∀k ∈ K

5. Binary and non-negativity:
   y_i ∈ {0,1},  ∀i ∈ I
   x_{ijk} ≥ 0,  ∀i,j,k
   z_{sik} ≥ 0,  ∀s,i,k
```

### Service-Constrained Warehouse Location

**Additional Parameters:**
- T_j: Maximum acceptable delivery time for customer j
- t_{ij}: Delivery time from warehouse i to customer j

**Service Constraint:**
```
Only serve customer j from warehouse i if delivery time acceptable:
x_{ijk} = 0  if  t_{ij} > T_j
```

Or as constraint:
```
x_{ijk} ≤ y_i × I(t_{ij} ≤ T_j),  ∀i,j,k

where I(condition) = 1 if condition true, 0 otherwise
```

---

## Solution Methods

### 1. MIP Model with PuLP

```python
from pulp import *
import numpy as np
import pandas as pd

def solve_warehouse_location(warehouse_data, customer_data,
                             transport_costs_out, transport_costs_in=None,
                             supplier_locations=None, products=None):
    """
    Solve multi-product warehouse location problem

    Args:
        warehouse_data: DataFrame with columns [warehouse_id, fixed_cost, capacity]
        customer_data: DataFrame with columns [customer_id, demand, lat, lon]
        transport_costs_out: cost matrix [warehouses x customers] or per-distance rate
        transport_costs_in: optional inbound costs
        supplier_locations: optional supplier data
        products: optional product list

    Returns:
        optimal solution
    """

    n_warehouses = len(warehouse_data)
    n_customers = len(customer_data)

    # Create problem
    prob = LpProblem("Warehouse_Location", LpMinimize)

    # Decision variables
    # y[i] = 1 if warehouse i is opened
    y = LpVariable.dicts("warehouse",
                         warehouse_data.index,
                         cat='Binary')

    # x[i,j] = fraction of customer j served by warehouse i
    x = LpVariable.dicts("service",
                         [(i, j) for i in warehouse_data.index
                          for j in customer_data.index],
                         lowBound=0, upBound=1, cat='Continuous')

    # Objective: Minimize total cost
    # Fixed costs
    fixed_cost_expr = lpSum([
        warehouse_data.loc[i, 'fixed_cost'] * y[i]
        for i in warehouse_data.index
    ])

    # Transportation costs
    transport_cost_expr = lpSum([
        transport_costs_out[i][j] * customer_data.loc[j, 'demand'] * x[i,j]
        for i in warehouse_data.index
        for j in customer_data.index
    ])

    prob += fixed_cost_expr + transport_cost_expr, "Total_Cost"

    # Constraints

    # 1. Each customer fully served
    for j in customer_data.index:
        prob += (
            lpSum([x[i,j] for i in warehouse_data.index]) == 1,
            f"Demand_Customer_{j}"
        )

    # 2. Warehouse capacity constraints
    for i in warehouse_data.index:
        prob += (
            lpSum([customer_data.loc[j, 'demand'] * x[i,j]
                   for j in customer_data.index]) <=
            warehouse_data.loc[i, 'capacity'] * y[i],
            f"Capacity_Warehouse_{i}"
        )

    # 3. Serve only from open warehouses
    for i in warehouse_data.index:
        for j in customer_data.index:
            prob += (
                x[i,j] <= y[i],
                f"Open_{i}_{j}"
            )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        open_warehouses = [i for i in warehouse_data.index
                          if y[i].varValue > 0.5]

        # Customer assignments
        assignments = {}
        for j in customer_data.index:
            assignments[j] = []
            for i in warehouse_data.index:
                if x[i,j].varValue > 0.01:
                    assignments[j].append({
                        'warehouse': i,
                        'fraction': x[i,j].varValue
                    })

        # Calculate warehouse utilization
        utilization = {}
        for i in open_warehouses:
            used_capacity = sum(
                customer_data.loc[j, 'demand'] * x[i,j].varValue
                for j in customer_data.index
            )
            utilization[i] = (used_capacity /
                            warehouse_data.loc[i, 'capacity'] * 100)

        # Cost breakdown
        total_fixed_cost = sum(
            warehouse_data.loc[i, 'fixed_cost']
            for i in open_warehouses
        )

        total_transport_cost = sum(
            transport_costs_out[i][j] *
            customer_data.loc[j, 'demand'] *
            x[i,j].varValue
            for i in warehouse_data.index
            for j in customer_data.index
        )

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'fixed_cost': total_fixed_cost,
            'transport_cost': total_transport_cost,
            'open_warehouses': open_warehouses,
            'num_warehouses': len(open_warehouses),
            'assignments': assignments,
            'utilization': utilization,
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # Warehouse data: 8 potential locations
    warehouse_data = pd.DataFrame({
        'warehouse_id': range(8),
        'fixed_cost': [500000, 450000, 600000, 520000,
                      480000, 550000, 490000, 530000],
        'capacity': [10000, 8000, 12000, 9000, 8500, 11000, 9500, 10500],
        'lat': np.random.uniform(30, 45, 8),
        'lon': np.random.uniform(-120, -70, 8)
    })
    warehouse_data.index = warehouse_data['warehouse_id']

    # Customer data: 30 customers
    customer_data = pd.DataFrame({
        'customer_id': range(30),
        'demand': np.random.uniform(100, 500, 30),
        'lat': np.random.uniform(30, 45, 30),
        'lon': np.random.uniform(-120, -70, 30)
    })
    customer_data.index = customer_data['customer_id']

    # Calculate transportation costs (simplified: Euclidean distance × rate)
    transport_rate = 0.5  # $ per unit per distance unit

    transport_costs_out = np.zeros((len(warehouse_data), len(customer_data)))
    for i in warehouse_data.index:
        for j in customer_data.index:
            distance = np.sqrt(
                (warehouse_data.loc[i, 'lat'] - customer_data.loc[j, 'lat'])**2 +
                (warehouse_data.loc[i, 'lon'] - customer_data.loc[j, 'lon'])**2
            )
            transport_costs_out[i][j] = distance * transport_rate

    print("="*70)
    print("WAREHOUSE LOCATION OPTIMIZATION")
    print("="*70)
    print(f"Potential warehouses: {len(warehouse_data)}")
    print(f"Customers: {len(customer_data)}")
    print(f"Total demand: {customer_data['demand'].sum():,.0f} units")

    # Solve
    result = solve_warehouse_location(warehouse_data, customer_data,
                                     transport_costs_out)

    print(f"\n{'='*70}")
    print(f"SOLUTION")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:,.2f}")
    print(f"  Fixed Costs: ${result['fixed_cost']:,.2f} "
          f"({result['fixed_cost']/result['total_cost']*100:.1f}%)")
    print(f"  Transport Costs: ${result['transport_cost']:,.2f} "
          f"({result['transport_cost']/result['total_cost']*100:.1f}%)")
    print(f"\nWarehouses Opened: {result['num_warehouses']}")
    print(f"Warehouse IDs: {result['open_warehouses']}")

    print(f"\nWarehouse Utilization:")
    for wh_id in result['open_warehouses']:
        capacity = warehouse_data.loc[wh_id, 'capacity']
        util = result['utilization'][wh_id]
        print(f"  Warehouse {wh_id}: {util:.1f}% (capacity={capacity:,.0f})")

    print(f"\nSolve Time: {result['solve_time']:.2f} seconds")
```

### 2. Gravity Location Model

```python
def gravity_location_model(customer_locations, customer_demands,
                          num_warehouses=None):
    """
    Gravity/Center-of-Gravity model for warehouse location

    Places warehouses at demand-weighted centroids

    Simple but effective heuristic for initial solutions

    Args:
        customer_locations: array of [lat, lon] for each customer
        customer_demands: customer demand weights
        num_warehouses: number of warehouses (if None, finds single location)

    Returns:
        optimal warehouse location(s)
    """
    from sklearn.cluster import KMeans

    customer_locations = np.array(customer_locations)
    customer_demands = np.array(customer_demands)

    if num_warehouses is None or num_warehouses == 1:
        # Single warehouse: weighted centroid
        total_demand = customer_demands.sum()

        center_lat = (customer_locations[:, 0] * customer_demands).sum() / total_demand
        center_lon = (customer_locations[:, 1] * customer_demands).sum() / total_demand

        return np.array([[center_lat, center_lon]])

    else:
        # Multiple warehouses: use weighted K-means clustering
        kmeans = KMeans(n_clusters=num_warehouses, random_state=42)

        # Weight samples by demand (repeat points proportional to demand)
        weights = (customer_demands / customer_demands.min()).astype(int)
        weighted_locations = []

        for i, loc in enumerate(customer_locations):
            weighted_locations.extend([loc] * weights[i])

        weighted_locations = np.array(weighted_locations)

        # Fit clustering
        kmeans.fit(weighted_locations)

        return kmeans.cluster_centers_


# Example usage
customer_locs = np.random.rand(50, 2) * 100  # 50 customers
customer_dems = np.random.uniform(100, 1000, 50)

# Find optimal locations for 3 warehouses
warehouse_locations = gravity_location_model(customer_locs, customer_dems,
                                            num_warehouses=3)

print("Optimal warehouse locations (gravity model):")
for i, loc in enumerate(warehouse_locations):
    print(f"  Warehouse {i+1}: ({loc[0]:.2f}, {loc[1]:.2f})")
```

### 3. Coverage-Based Location

```python
def coverage_based_warehouse_location(customer_locations, service_radius,
                                     potential_warehouses=None):
    """
    Warehouse location with service coverage constraints

    Ensure all customers within service radius of at least one warehouse

    Args:
        customer_locations: customer coordinates
        service_radius: maximum service distance
        potential_warehouses: candidate warehouse locations
                            (if None, use customer locations)

    Returns:
        minimum set of warehouses for full coverage
    """

    if potential_warehouses is None:
        potential_warehouses = customer_locations

    n_warehouses = len(potential_warehouses)
    n_customers = len(customer_locations)

    # Calculate coverage matrix
    # coverage[i][j] = 1 if warehouse i can serve customer j
    coverage = np.zeros((n_warehouses, n_customers))

    for i in range(n_warehouses):
        for j in range(n_customers):
            distance = np.linalg.norm(
                potential_warehouses[i] - customer_locations[j]
            )
            if distance <= service_radius:
                coverage[i][j] = 1

    # Solve set covering problem
    prob = LpProblem("Coverage_Warehouse_Location", LpMinimize)

    # Decision variables: y[i] = 1 if warehouse i is opened
    y = LpVariable.dicts("warehouse", range(n_warehouses), cat='Binary')

    # Objective: Minimize number of warehouses
    prob += lpSum([y[i] for i in range(n_warehouses)]), "Num_Warehouses"

    # Constraints: Each customer covered by at least one warehouse
    for j in range(n_customers):
        prob += (
            lpSum([coverage[i][j] * y[i] for i in range(n_warehouses)]) >= 1,
            f"Coverage_Customer_{j}"
        )

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        open_warehouses = [i for i in range(n_warehouses)
                          if y[i].varValue > 0.5]

        # Determine which warehouse serves each customer
        assignments = {}
        for j in range(n_customers):
            # Assign to nearest warehouse that can serve
            min_distance = float('inf')
            assigned_warehouse = None

            for i in open_warehouses:
                if coverage[i][j] == 1:
                    distance = np.linalg.norm(
                        potential_warehouses[i] - customer_locations[j]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        assigned_warehouse = i

            assignments[j] = assigned_warehouse

        return {
            'status': LpStatus[prob.status],
            'num_warehouses': len(open_warehouses),
            'open_warehouses': open_warehouses,
            'warehouse_locations': [potential_warehouses[i]
                                  for i in open_warehouses],
            'assignments': assignments
        }

    return {'status': LpStatus[prob.status]}


# Example
customer_locs = np.random.rand(40, 2) * 100
potential_wh_locs = np.random.rand(15, 2) * 100
service_radius = 25

result = coverage_based_warehouse_location(customer_locs, service_radius,
                                          potential_wh_locs)

print(f"Minimum warehouses for coverage: {result['num_warehouses']}")
print(f"Warehouse indices: {result['open_warehouses']}")
```

---

## Heuristic Approaches

### 1. Greedy Opening Heuristic

```python
def greedy_warehouse_opening(warehouse_costs, customer_demands,
                            transport_costs, capacities):
    """
    Greedy heuristic: open warehouses one at a time

    Select warehouse that gives maximum cost reduction

    Args:
        warehouse_costs: fixed costs for each warehouse
        customer_demands: customer demands
        transport_costs: matrix [warehouses x customers]
        capacities: warehouse capacities

    Returns:
        heuristic solution
    """
    n_warehouses = len(warehouse_costs)
    n_customers = len(customer_demands)

    open_warehouses = []
    unserved_customers = set(range(n_customers))

    def calculate_total_cost(open_whs, assignments):
        """Calculate total cost for given configuration"""
        fixed = sum(warehouse_costs[i] for i in open_whs)
        transport = sum(
            transport_costs[assignments[j]][j] * customer_demands[j]
            for j in range(n_customers) if j in assignments
        )
        return fixed + transport

    # Iteratively open warehouses
    while unserved_customers:
        best_warehouse = None
        best_cost = float('inf')
        best_assignments = None

        # Try opening each unopened warehouse
        for wh in range(n_warehouses):
            if wh in open_warehouses:
                continue

            test_warehouses = open_warehouses + [wh]

            # Assign customers greedily to nearest warehouse
            test_assignments = {}
            remaining_capacity = {w: capacities[w] for w in test_warehouses}

            # Sort customers by closest distance to this new warehouse
            customer_distances = [
                (transport_costs[wh][j], j) for j in unserved_customers
            ]
            customer_distances.sort()

            for dist, cust in customer_distances:
                # Assign to nearest warehouse with capacity
                assigned = False
                for w in sorted(test_warehouses,
                              key=lambda x: transport_costs[x][cust]):
                    if remaining_capacity[w] >= customer_demands[cust]:
                        test_assignments[cust] = w
                        remaining_capacity[w] -= customer_demands[cust]
                        assigned = True
                        break

                if not assigned:
                    test_assignments[cust] = min(test_warehouses,
                                                key=lambda x: transport_costs[x][cust])

            # Calculate cost
            test_cost = calculate_total_cost(test_warehouses, test_assignments)

            if test_cost < best_cost:
                best_cost = test_cost
                best_warehouse = wh
                best_assignments = test_assignments

        # Open best warehouse
        if best_warehouse is not None:
            open_warehouses.append(best_warehouse)
            unserved_customers = {c for c in unserved_customers
                                if c not in best_assignments}

            # Update assignments for served customers
            if not unserved_customers:
                assignments = best_assignments
                break
        else:
            break

    total_cost = calculate_total_cost(open_warehouses, assignments)

    return {
        'open_warehouses': open_warehouses,
        'assignments': assignments,
        'total_cost': total_cost,
        'method': 'Greedy Opening'
    }
```

### 2. Savings-Based Consolidation

```python
def savings_based_consolidation(warehouse_costs, customer_demands,
                               transport_costs, initial_warehouses=None):
    """
    Savings-based heuristic for warehouse consolidation

    Start with many warehouses, consolidate based on savings

    Args:
        warehouse_costs: fixed costs
        customer_demands: demands
        transport_costs: transport cost matrix
        initial_warehouses: starting warehouse set

    Returns:
        consolidated solution
    """
    n_warehouses = len(warehouse_costs)
    n_customers = len(customer_demands)

    # Start with all warehouses if not specified
    if initial_warehouses is None:
        open_warehouses = set(range(n_warehouses))
    else:
        open_warehouses = set(initial_warehouses)

    def assign_customers(whs):
        """Assign each customer to nearest warehouse"""
        assignments = {}
        for j in range(n_customers):
            nearest = min(whs, key=lambda i: transport_costs[i][j])
            assignments[j] = nearest
        return assignments

    def calculate_cost(whs, assignments):
        """Calculate total cost"""
        fixed = sum(warehouse_costs[i] for i in whs)
        transport = sum(
            transport_costs[assignments[j]][j] * customer_demands[j]
            for j in range(n_customers)
        )
        return fixed + transport

    improved = True
    while improved and len(open_warehouses) > 1:
        improved = False

        current_assignments = assign_customers(open_warehouses)
        current_cost = calculate_cost(open_warehouses, current_assignments)

        # Try closing each warehouse
        best_warehouse_to_close = None
        best_cost_after_closing = current_cost

        for wh in list(open_warehouses):
            # Test closing this warehouse
            test_warehouses = open_warehouses - {wh}
            test_assignments = assign_customers(test_warehouses)
            test_cost = calculate_cost(test_warehouses, test_assignments)

            # Check if closing reduces cost
            if test_cost < best_cost_after_closing:
                best_cost_after_closing = test_cost
                best_warehouse_to_close = wh
                improved = True

        # Close best warehouse if improvement found
        if improved:
            open_warehouses.remove(best_warehouse_to_close)

    final_assignments = assign_customers(open_warehouses)
    final_cost = calculate_cost(open_warehouses, final_assignments)

    return {
        'open_warehouses': list(open_warehouses),
        'assignments': final_assignments,
        'total_cost': final_cost,
        'method': 'Savings-Based Consolidation'
    }
```

---

## Complete Warehouse Location Solver

```python
class WarehouseLocationSolver:
    """
    Comprehensive warehouse location optimization solver
    """

    def __init__(self):
        self.warehouse_data = None
        self.customer_data = None
        self.transport_costs = None
        self.solution = None

    def load_problem(self, warehouse_data, customer_data,
                    transport_costs=None, transport_rate=None):
        """
        Load problem data

        Args:
            warehouse_data: DataFrame with warehouse info
            customer_data: DataFrame with customer info
            transport_costs: cost matrix or None
            transport_rate: if transport_costs None, calculate from rate
        """
        self.warehouse_data = warehouse_data
        self.customer_data = customer_data

        if transport_costs is not None:
            self.transport_costs = transport_costs
        elif transport_rate is not None:
            # Calculate costs from distances
            self.transport_costs = self._calculate_transport_costs(transport_rate)
        else:
            raise ValueError("Must provide transport_costs or transport_rate")

        print(f"Loaded warehouse location problem:")
        print(f"  Potential warehouses: {len(warehouse_data)}")
        print(f"  Customers: {len(customer_data)}")
        print(f"  Total demand: {customer_data['demand'].sum():,.0f}")

    def _calculate_transport_costs(self, rate):
        """Calculate transport costs from coordinates and rate"""
        costs = np.zeros((len(self.warehouse_data), len(self.customer_data)))

        for i in self.warehouse_data.index:
            for j in self.customer_data.index:
                dist = np.sqrt(
                    (self.warehouse_data.loc[i, 'lat'] -
                     self.customer_data.loc[j, 'lat'])**2 +
                    (self.warehouse_data.loc[i, 'lon'] -
                     self.customer_data.loc[j, 'lon'])**2
                )
                costs[i][j] = dist * rate

        return costs

    def solve_exact(self, time_limit=600):
        """Solve with MIP (exact)"""
        print("\nSolving with MIP (exact method)...")
        return solve_warehouse_location(
            self.warehouse_data,
            self.customer_data,
            self.transport_costs
        )

    def solve_heuristic(self, method='greedy'):
        """
        Solve with heuristic

        Args:
            method: 'greedy', 'gravity', 'coverage', 'savings'
        """
        print(f"\nSolving with {method} heuristic...")

        if method == 'greedy':
            return greedy_warehouse_opening(
                self.warehouse_data['fixed_cost'].values,
                self.customer_data['demand'].values,
                self.transport_costs,
                self.warehouse_data['capacity'].values
            )

        elif method == 'gravity':
            customer_locs = self.customer_data[['lat', 'lon']].values
            customer_dems = self.customer_data['demand'].values

            # Estimate number of warehouses from capacity
            total_demand = customer_dems.sum()
            avg_capacity = self.warehouse_data['capacity'].mean()
            num_whs = max(1, int(np.ceil(total_demand / avg_capacity * 1.2)))

            locations = gravity_location_model(customer_locs, customer_dems,
                                             num_warehouses=num_whs)

            # Match to nearest potential warehouses
            open_whs = []
            for loc in locations:
                nearest = None
                min_dist = float('inf')
                for i in self.warehouse_data.index:
                    wh_loc = self.warehouse_data.loc[i, ['lat', 'lon']].values
                    dist = np.linalg.norm(loc - wh_loc)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
                open_whs.append(nearest)

            return {'open_warehouses': open_whs, 'method': 'Gravity'}

        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_solutions(self, methods=['greedy', 'savings', 'exact']):
        """Compare multiple solution methods"""
        import pandas as pd
        import time

        results = []

        for method in methods:
            start_time = time.time()

            try:
                if method == 'exact':
                    solution = self.solve_exact()
                else:
                    solution = self.solve_heuristic(method)

                solve_time = time.time() - start_time

                results.append({
                    'Method': method,
                    'Total Cost': solution['total_cost'],
                    'Warehouses': len(solution['open_warehouses']),
                    'Time (s)': f"{solve_time:.2f}"
                })

            except Exception as e:
                print(f"Error with {method}: {e}")

        df = pd.DataFrame(results)

        if len(df) > 0:
            best_cost = df['Total Cost'].min()
            df['Gap %'] = ((df['Total Cost'] - best_cost) / best_cost * 100).round(2)

        return df

    def visualize_network(self, solution, title="Warehouse Network"):
        """Visualize warehouse network"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 10))

        # Plot customers (blue circles)
        plt.scatter(self.customer_data['lon'],
                   self.customer_data['lat'],
                   c='lightblue', s=50, alpha=0.6,
                   label='Customers')

        # Plot all potential warehouses (gray)
        plt.scatter(self.warehouse_data['lon'],
                   self.warehouse_data['lat'],
                   c='lightgray', s=200, alpha=0.3,
                   marker='s', label='Potential Warehouses')

        # Plot open warehouses (red)
        open_wh_data = self.warehouse_data.loc[solution['open_warehouses']]
        plt.scatter(open_wh_data['lon'],
                   open_wh_data['lat'],
                   c='red', s=300, alpha=0.8,
                   marker='s', label='Open Warehouses',
                   edgecolors='black', linewidths=2)

        # Draw assignments
        if 'assignments' in solution:
            for cust_id, assignment in solution['assignments'].items():
                if isinstance(assignment, list):
                    wh_id = assignment[0]['warehouse']
                else:
                    wh_id = assignment

                cust_loc = self.customer_data.loc[cust_id, ['lon', 'lat']].values
                wh_loc = self.warehouse_data.loc[wh_id, ['lon', 'lat']].values

                plt.plot([cust_loc[0], wh_loc[0]],
                        [cust_loc[1], wh_loc[1]],
                        'k-', alpha=0.1, linewidth=0.5)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Complete example
if __name__ == "__main__":
    print("="*70)
    print("WAREHOUSE LOCATION OPTIMIZATION - COMPLETE EXAMPLE")
    print("="*70)

    np.random.seed(42)

    # Generate problem data
    n_warehouses = 10
    n_customers = 50

    warehouse_df = pd.DataFrame({
        'warehouse_id': range(n_warehouses),
        'fixed_cost': np.random.uniform(400000, 700000, n_warehouses),
        'capacity': np.random.uniform(5000, 15000, n_warehouses),
        'lat': np.random.uniform(30, 45, n_warehouses),
        'lon': np.random.uniform(-120, -70, n_warehouses)
    })
    warehouse_df.index = warehouse_df['warehouse_id']

    customer_df = pd.DataFrame({
        'customer_id': range(n_customers),
        'demand': np.random.uniform(100, 800, n_customers),
        'lat': np.random.uniform(30, 45, n_customers),
        'lon': np.random.uniform(-120, -70, n_customers)
    })
    customer_df.index = customer_df['customer_id']

    # Create solver
    solver = WarehouseLocationSolver()
    solver.load_problem(warehouse_df, customer_df, transport_rate=0.5)

    # Compare solutions
    print("\n" + "="*70)
    print("COMPARING SOLUTION METHODS")
    print("="*70)

    comparison = solver.compare_solutions(['greedy', 'exact'])
    print("\n" + comparison.to_string(index=False))

    # Detailed solution
    print("\n" + "="*70)
    print("DETAILED OPTIMAL SOLUTION")
    print("="*70)

    best_solution = solver.solve_exact()

    print(f"Total Cost: ${best_solution['total_cost']:,.2f}")
    print(f"  Fixed: ${best_solution['fixed_cost']:,.2f} "
          f"({best_solution['fixed_cost']/best_solution['total_cost']*100:.1f}%)")
    print(f"  Transport: ${best_solution['transport_cost']:,.2f} "
          f"({best_solution['transport_cost']/best_solution['total_cost']*100:.1f}%)")

    print(f"\nWarehouses Opened: {best_solution['num_warehouses']}")
    for wh_id in best_solution['open_warehouses']:
        util = best_solution['utilization'][wh_id]
        cap = warehouse_df.loc[wh_id, 'capacity']
        cost = warehouse_df.loc[wh_id, 'fixed_cost']
        print(f"  Warehouse {wh_id}: Util={util:.1f}%, "
              f"Cap={cap:,.0f}, Cost=${cost:,.0f}")

    # Visualize
    solver.visualize_network(best_solution,
                            title="Optimal Warehouse Network")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **PuLP**: MIP modeling
- **Pyomo**: Advanced optimization
- **OR-Tools**: Google optimization
- **Gurobi/CPLEX**: Commercial solvers

**Data & Analysis:**
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Clustering, ML

**Geospatial:**
- **geopy**: Distance calculations
- **folium**: Interactive maps
- **geopandas**: Geospatial data

**Visualization:**
- **matplotlib**: Plotting
- **plotly**: Interactive viz
- **seaborn**: Statistical viz

### Commercial Software

- **Llamasoft (Coupa)**: Supply chain network design
- **LLamasoft Design**: Warehouse optimization
- **SAP IBP**: Integrated business planning
- **Blue Yonder**: Supply chain platform
- **o9 Solutions**: Planning platform

---

## Common Challenges & Solutions

### Challenge: Multi-Echelon Networks

**Problem:**
- Plants → Regional DCs → Local warehouses → Customers
- Complex flow patterns
- Multiple decisions layers

**Solutions:**
- Hierarchical decomposition
- Multi-stage optimization
- Simultaneous optimization if tractable
- Iterative refinement

### Challenge: Seasonal Demand

**Problem:**
- Demand varies significantly by season
- Warehouse decisions long-term
- Peak capacity requirements

**Solutions:**
- Model peak season explicitly
- Include flexibility/surge capacity
- Temporary warehouses for peak
- Third-party logistics (3PL) options
- Multi-period models

### Challenge: Inventory Considerations

**Problem:**
- Warehouse location affects inventory levels
- More warehouses → more safety stock
- Trade-off not captured in simple models

**Solutions:**
- Include inventory costs in model
- Square root law for safety stock
- Risk pooling benefits of centralization
- Multi-echelon inventory optimization

### Challenge: Real-World Constraints

**Problem:**
- Zoning regulations
- Labor availability
- Real estate availability
- Infrastructure (ports, rails, highways)
- Tax incentives

**Solutions:**
- Include as constraints in model
- Scenario analysis
- Post-optimization feasibility checks
- Collaboration with site selection experts

---

## Output Format

### Warehouse Location Solution Report

**Problem Instance:**
- Candidate Warehouses: 15
- Customers: 100
- Total Annual Demand: 125,000 units
- Planning Horizon: 5 years

**Optimal Network Configuration:**

| Metric | Value |
|--------|-------|
| Total Annual Cost | $8,247,500 |
| Fixed Costs | $2,450,000 (29.7%) |
| Outbound Transport | $4,823,000 (58.5%) |
| Inbound Transport | $974,500 (11.8%) |
| Warehouses Opened | 4 |
| Average Utilization | 78.3% |
| Service Coverage | 100% |

**Open Warehouses:**

| WH ID | Location | Fixed Cost | Capacity | Utilization | Customers Served |
|-------|----------|------------|----------|-------------|------------------|
| 3 | Dallas, TX | $650,000 | 35,000 | 82.1% | 28 |
| 7 | Atlanta, GA | $580,000 | 30,000 | 76.8% | 24 |
| 11 | Los Angeles, CA | $720,000 | 40,000 | 81.4% | 31 |
| 14 | Chicago, IL | $500,000 | 25,000 | 72.9% | 17 |

**Network Statistics:**
- Average distance to customer: 287 miles
- Maximum distance to customer: 612 miles
- Average outbound transport cost per unit: $38.58
- Capacity buffer: 21.7% (for growth/seasonality)

---

## Questions to Ask

1. Is this a greenfield (new network) or brownfield (existing) analysis?
2. How many potential warehouse locations?
3. How many customers? What are their locations and demands?
4. What are the warehouse fixed costs (lease/construction)?
5. Operating costs (labor, utilities, management)?
6. Warehouse capacities or capacity options?
7. What are transportation costs? (rates or distance-based)
8. Service requirements? (delivery time, coverage distance)
9. Are there existing facilities that must remain?
10. Planning horizon? (1 year, 3 years, 10 years)
11. Demand variability and growth projections?
12. Multi-echelon network? (plants, DCs, local warehouses)
13. Product characteristics? (value, weight, storage requirements)
14. Inventory holding costs important?

---

## Related Skills

- **facility-location-problem**: General facility location theory
- **distribution-center-network**: DC-specific network design
- **network-design**: End-to-end supply chain network
- **hub-location-problem**: Hub-and-spoke networks
- **set-covering-problem**: Coverage-based location
- **inventory-optimization**: Inventory-location trade-offs
- **network-flow-optimization**: Flow allocation in networks
- **vehicle-routing-problem**: Last-mile delivery from warehouses
- **multi-echelon-inventory**: Inventory across network levels

---
name: vrp-time-windows
description: When the user wants to solve VRP with time windows (VRPTW), optimize routes with delivery time constraints, or handle appointment scheduling. Also use when the user mentions "VRPTW," "time window routing," "scheduled deliveries," "appointment routing," "delivery windows," "earliest/latest delivery," or "hard/soft time windows." For basic VRP, see vehicle-routing-problem.
---

# Vehicle Routing Problem with Time Windows (VRPTW)

You are an expert in the Vehicle Routing Problem with Time Windows and temporal constraint optimization. Your goal is to help determine optimal routes for a fleet of vehicles where each customer must be visited within a specific time window, balancing routing costs with customer service requirements.

## Initial Assessment

Before solving VRPTW instances, understand:

1. **Time Window Characteristics**
   - Hard time windows (must be satisfied) or soft (can violate with penalty)?
   - How many customers have time windows? All or subset?
   - Width of time windows? (narrow = harder problem)
   - Distribution of windows throughout the day?

2. **Temporal Parameters**
   - Service time at each customer?
   - Travel times between locations?
   - Vehicle shift duration/maximum route time?
   - Depot operating hours?
   - Driver break requirements?

3. **Fleet Information**
   - Number of vehicles available?
   - Vehicle capacities?
   - Earliest start time from depot?
   - Latest return time to depot?

4. **Problem Scale**
   - Small (< 25 customers): Exact methods possible
   - Medium (25-100 customers): Advanced heuristics
   - Large (100+ customers): Metaheuristics required

5. **Objectives**
   - Minimize total distance/time?
   - Minimize number of vehicles (primary)?
   - Minimize time window violations?
   - Minimize waiting time?

---

## Mathematical Formulation

### VRPTW with Hard Time Windows

**Sets:**
- V = {0, 1, ..., n}: Nodes (0 = depot, 1..n = customers)
- K = {1, ..., m}: Vehicles

**Parameters:**
- c_{ij}: Cost/distance from node i to j
- t_{ij}: Travel time from node i to j
- d_i: Demand at customer i
- s_i: Service time at customer i
- [e_i, l_i]: Time window at customer i (earliest, latest)
- Q_k: Capacity of vehicle k
- T_max: Maximum route duration

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j
- w_i ≥ 0: Arrival time at customer i

**Objective Function:**
```
Minimize: Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

Or minimize vehicles first:
```
Minimize: Σ_{k∈K} Σ_{j∈V\{0}} x_{0jk} + α * Σ_{k∈K} Σ_{i,j∈V} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each customer visited exactly once:
   Σ_{k∈K} Σ_{i∈V} x_{ijk} = 1,  ∀j ∈ V\{0}

2. Flow conservation:
   Σ_{i∈V} x_{ihk} - Σ_{j∈V} x_{hjk} = 0,  ∀h ∈ V, ∀k ∈ K

3. Vehicle starts from depot:
   Σ_{j∈V\{0}} x_{0jk} ≤ 1,  ∀k ∈ K

4. Vehicle returns to depot:
   Σ_{i∈V\{0}} x_{i0k} ≤ 1,  ∀k ∈ K

5. Capacity constraint:
   Σ_{i∈V\{0}} Σ_{j∈V} d_i * x_{ijk} ≤ Q_k,  ∀k ∈ K

6. Time window constraints:
   e_i ≤ w_i ≤ l_i,  ∀i ∈ V

7. Time consistency (if vehicle k goes from i to j):
   w_i + s_i + t_{ij} ≤ w_j + M*(1 - Σ_k x_{ijk}),  ∀i,j ∈ V, i≠j

8. Maximum route duration:
   w_i + s_i + t_{i0} ≤ T_max,  ∀i ∈ V\{0}

9. Binary variables:
   x_{ijk} ∈ {0,1},  ∀i,j ∈ V, ∀k ∈ K
```

### Soft Time Windows Formulation

Add penalty variables and costs:

**Additional Variables:**
- α_i ≥ 0: Early arrival at i (before e_i)
- β_i ≥ 0: Late arrival at i (after l_i)

**Modified Objective:**
```
Minimize: Σ_{k,i,j} c_{ij} * x_{ijk} +
          Σ_i (penalty_early * α_i + penalty_late * β_i)
```

**Modified Time Window Constraints:**
```
w_i + α_i ≥ e_i,  ∀i
w_i - β_i ≤ l_i,  ∀i
```

---

## Exact Algorithms

### 1. Branch-and-Price for VRPTW

```python
from pulp import *
import numpy as np

def vrptw_mip(dist_matrix, time_matrix, demands, time_windows,
              service_times, vehicle_capacity, num_vehicles,
              depot=0, max_route_time=480):
    """
    VRPTW using MIP formulation

    Args:
        dist_matrix: n x n distance matrix
        time_matrix: n x n travel time matrix (minutes)
        demands: list of demands
        time_windows: list of (earliest, latest) tuples (minutes)
        service_times: list of service times (minutes)
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot: depot index
        max_route_time: maximum route duration (minutes)

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Create problem
    prob = LpProblem("VRPTW", LpMinimize)

    # Decision variables
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    x[i,j,k] = LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    # Arrival time variables
    w = {}
    for i in range(n):
        for k in range(num_vehicles):
            w[i,k] = LpVariable(f"w_{i}_{k}",
                               lowBound=time_windows[i][0],
                               upBound=time_windows[i][1],
                               cat='Continuous')

    # Objective: Minimize total distance
    prob += lpSum([dist_matrix[i][j] * x[i,j,k]
                   for i in range(n) for j in range(n) if i != j
                   for k in range(num_vehicles)]), "Total_Distance"

    # Constraints

    # 1. Each customer visited exactly once
    for j in customers:
        prob += lpSum([x[i,j,k] for i in range(n) if i != j
                      for k in range(num_vehicles)]) == 1, f"Visit_{j}"

    # 2. Flow conservation
    for h in range(n):
        for k in range(num_vehicles):
            prob += (lpSum([x[i,h,k] for i in range(n) if i != h]) ==
                    lpSum([x[h,j,k] for j in range(n) if j != h])), \
                    f"Flow_{h}_{k}"

    # 3. Capacity constraints
    for k in range(num_vehicles):
        prob += lpSum([demands[j] * x[i,j,k]
                      for i in range(n) for j in customers if i != j]) \
                <= vehicle_capacity, f"Capacity_{k}"

    # 4. Time consistency constraints
    M = 10000  # Big-M
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    prob += (w[i,k] + service_times[i] + time_matrix[i][j] <=
                            w[j,k] + M * (1 - x[i,j,k])), \
                            f"Time_{i}_{j}_{k}"

    # 5. Time windows already enforced by variable bounds

    # 6. Maximum route duration
    for k in range(num_vehicles):
        for i in customers:
            prob += (w[i,k] + service_times[i] + time_matrix[i][depot] <=
                    time_windows[depot][1]), f"MaxTime_{i}_{k}"

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        routes = []
        arrival_times = []

        for k in range(num_vehicles):
            route = [depot]
            times = [w[depot,k].varValue]
            current = depot

            while True:
                next_node = None
                for j in range(n):
                    if j != current and (current,j,k) in x:
                        if x[current,j,k].varValue > 0.5:
                            next_node = j
                            break

                if next_node is None or next_node == depot:
                    route.append(depot)
                    times.append(w[depot,k].varValue +
                               sum(service_times[route[i]] + time_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1)))
                    break

                route.append(next_node)
                times.append(w[next_node,k].varValue)
                current = next_node

            if len(route) > 2:
                routes.append(route)
                arrival_times.append(times)

        return {
            'status': LpStatus[prob.status],
            'total_distance': value(prob.objective),
            'routes': routes,
            'arrival_times': arrival_times,
            'num_vehicles_used': len(routes),
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'total_distance': None,
            'routes': None,
            'solve_time': solve_time
        }
```

---

## Constructive Heuristics

### 1. Solomon's I1 Insertion Heuristic

```python
def solomon_i1_insertion(dist_matrix, time_matrix, demands, time_windows,
                        service_times, vehicle_capacity, depot=0,
                        alpha=1.0, mu=1.0, lambda_param=1.0):
    """
    Solomon's I1 insertion heuristic for VRPTW

    One of the best constructive heuristics for VRPTW

    Args:
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times
        vehicle_capacity: vehicle capacity
        depot: depot index
        alpha, mu, lambda_param: criterion weights

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    customers = set(range(n)) - {depot}
    routes = []
    route_loads = []
    route_times = []

    while customers:
        # Initialize new route with seed customer
        # Select farthest customer from depot
        seed = max(customers, key=lambda c: dist_matrix[depot][c])

        route = [depot, seed, depot]
        current_time = [time_windows[depot][0],
                       max(time_windows[depot][0] + time_matrix[depot][seed],
                           time_windows[seed][0]),
                       0]

        # Update current_time[2] (return to depot)
        current_time[2] = (current_time[1] + service_times[seed] +
                          time_matrix[seed][depot])

        current_load = demands[seed]
        customers.remove(seed)

        # Insert remaining customers into this route
        while customers:
            best_customer = None
            best_position = None
            best_criterion = float('inf')

            # Try inserting each unrouted customer
            for customer in list(customers):
                if current_load + demands[customer] > vehicle_capacity:
                    continue

                # Try each insertion position
                for pos in range(1, len(route)):
                    i = route[pos - 1]
                    j = route[pos]

                    # Calculate feasibility
                    # Arrival time at customer
                    arrival_at_customer = (current_time[pos-1] +
                                         service_times[i] +
                                         time_matrix[i][customer])

                    # Check time window feasibility
                    if arrival_at_customer > time_windows[customer][1]:
                        continue  # Too late

                    # Start service time (wait if early)
                    start_service = max(arrival_at_customer,
                                      time_windows[customer][0])

                    # Check if rest of route is still feasible
                    push_forward = max(0, start_service + service_times[customer] +
                                     time_matrix[customer][j] - current_time[pos])

                    # Check if pushing forward violates time windows
                    feasible = True
                    temp_time = current_time[pos] + push_forward

                    for k in range(pos, len(route) - 1):
                        if temp_time > time_windows[route[k]][1]:
                            feasible = False
                            break
                        temp_time = (max(temp_time, time_windows[route[k]][0]) +
                                   service_times[route[k]] +
                                   time_matrix[route[k]][route[k+1]])

                    if not feasible:
                        continue

                    # Calculate insertion criterion (c1)
                    # c11: distance increase
                    c11 = (dist_matrix[i][customer] + dist_matrix[customer][j] -
                          mu * dist_matrix[i][j])

                    # c12: time increase
                    c12 = (start_service - current_time[pos-1])

                    c1 = alpha * c11 + (1 - alpha) * c12

                    # c2: distance from depot (encourages early insertion)
                    c2 = dist_matrix[depot][customer]

                    # Combined criterion
                    criterion = lambda_param * c1 - c2

                    if criterion < best_criterion:
                        best_criterion = criterion
                        best_customer = customer
                        best_position = pos

            if best_customer is None:
                break  # No more customers fit

            # Insert best customer
            route.insert(best_position, best_customer)
            current_load += demands[best_customer]

            # Update arrival times
            new_times = [current_time[0]]
            for k in range(1, len(route)):
                arrival = (new_times[k-1] +
                          service_times[route[k-1]] +
                          time_matrix[route[k-1]][route[k]])
                start = max(arrival, time_windows[route[k]][0])
                new_times.append(start)

            current_time = new_times
            customers.remove(best_customer)

        routes.append(route)
        route_loads.append(current_load)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes
    )

    return {
        'routes': routes,
        'route_loads': route_loads,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

### 2. Nearest Neighbor with Time Windows

```python
def nearest_neighbor_tw(dist_matrix, time_matrix, demands, time_windows,
                       service_times, vehicle_capacity, depot=0):
    """
    Nearest neighbor heuristic adapted for time windows

    Args:
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times
        vehicle_capacity: vehicle capacity
        depot: depot index

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    unrouted = set(range(n)) - {depot}
    routes = []

    while unrouted:
        route = [depot]
        current_time = time_windows[depot][0]
        current_load = 0
        current_location = depot

        while True:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')

            for customer in unrouted:
                # Check capacity
                if current_load + demands[customer] > vehicle_capacity:
                    continue

                # Check time window feasibility
                arrival_time = current_time + time_matrix[current_location][customer]

                if arrival_time > time_windows[customer][1]:
                    continue  # Too late

                # This customer is feasible
                distance = dist_matrix[current_location][customer]

                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer

            if best_customer is None:
                break  # No feasible customers

            # Add customer to route
            route.append(best_customer)
            arrival_time = current_time + time_matrix[current_location][best_customer]
            current_time = max(arrival_time, time_windows[best_customer][0])
            current_time += service_times[best_customer]
            current_load += demands[best_customer]
            current_location = best_customer
            unrouted.remove(best_customer)

        # Return to depot
        route.append(depot)
        routes.append(route)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes
    )

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

---

## Improvement Heuristics

### 1. 2-Opt with Time Window Feasibility

```python
def two_opt_tw(route, dist_matrix, time_matrix, time_windows, service_times):
    """
    2-opt improvement with time window checks

    Args:
        route: route to improve
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        time_windows: time windows
        service_times: service times

    Returns:
        improved route
    """
    def check_tw_feasibility(route):
        """Check if route satisfies all time windows"""
        current_time = time_windows[route[0]][0]

        for i in range(len(route) - 1):
            current_time += time_matrix[route[i]][route[i+1]]

            # Check if arrival is within time window
            if current_time > time_windows[route[i+1]][1]:
                return False

            # Wait if early
            current_time = max(current_time, time_windows[route[i+1]][0])
            current_time += service_times[route[i+1]]

        return True

    improved = True
    best_route = route.copy()

    while improved:
        improved = False
        n = len(best_route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Try reversing segment [i, j]
                new_route = best_route.copy()
                new_route[i:j+1] = reversed(new_route[i:j+1])

                # Check time window feasibility
                if not check_tw_feasibility(new_route):
                    continue

                # Calculate distance change
                old_dist = (dist_matrix[best_route[i-1]][best_route[i]] +
                           dist_matrix[best_route[j]][best_route[j+1]])
                new_dist = (dist_matrix[new_route[i-1]][new_route[i]] +
                           dist_matrix[new_route[j]][new_route[j+1]])

                if new_dist < old_dist - 1e-10:
                    best_route = new_route
                    improved = True
                    break

            if improved:
                break

    return best_route
```

### 2. Cross-Exchange with Time Windows

```python
def cross_exchange_tw(routes, dist_matrix, time_matrix, demands,
                     time_windows, service_times, vehicle_capacity):
    """
    Cross-exchange operator with time window feasibility

    Args:
        routes: list of routes
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: time windows
        service_times: service times
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    def calculate_route_cost(route):
        return sum(dist_matrix[route[i]][route[i+1]]
                  for i in range(len(route)-1))

    def check_route_feasibility(route):
        """Check capacity and time window feasibility"""
        # Check capacity
        load = sum(demands[c] for c in route[1:-1])
        if load > vehicle_capacity:
            return False

        # Check time windows
        current_time = time_windows[route[0]][0]
        for i in range(len(route) - 1):
            current_time += time_matrix[route[i]][route[i+1]]
            if current_time > time_windows[route[i+1]][1]:
                return False
            current_time = max(current_time, time_windows[route[i+1]][0])
            current_time += service_times[route[i+1]]

        return True

    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(r1 + 1, num_routes):
                route1 = routes[r1].copy()
                route2 = routes[r2].copy()

                # Try swapping customers
                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Create new routes with swap
                        new_route1 = route1.copy()
                        new_route2 = route2.copy()

                        new_route1[i] = route2[j]
                        new_route2[j] = route1[i]

                        # Check feasibility
                        if (not check_route_feasibility(new_route1) or
                            not check_route_feasibility(new_route2)):
                            continue

                        # Calculate improvement
                        old_cost = (calculate_route_cost(route1) +
                                  calculate_route_cost(route2))
                        new_cost = (calculate_route_cost(new_route1) +
                                  calculate_route_cost(new_route2))

                        if new_cost < old_cost - 1e-10:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes
```

---

## Metaheuristics

### 1. Adaptive Large Neighborhood Search (ALNS)

```python
import random

def alns_vrptw(dist_matrix, time_matrix, demands, time_windows,
              service_times, vehicle_capacity, initial_solution,
              iterations=1000, destroy_size=0.3):
    """
    Adaptive Large Neighborhood Search for VRPTW

    Args:
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: time windows
        service_times: service times
        vehicle_capacity: vehicle capacity
        initial_solution: initial routes
        iterations: number of iterations
        destroy_size: fraction of customers to remove

    Returns:
        best solution found
    """
    import copy

    def calculate_cost(routes):
        return sum(sum(dist_matrix[route[i]][route[i+1]]
                      for i in range(len(route)-1))
                  for route in routes)

    def shaw_removal_tw(routes, num_remove):
        """Remove related customers (considering location and time)"""
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])

        if not all_customers or num_remove == 0:
            return routes, []

        seed = random.choice(all_customers)
        to_remove = {seed}

        # Calculate relatedness (distance + time window similarity)
        relatedness = []
        for customer in all_customers:
            if customer != seed:
                dist_similarity = dist_matrix[seed][customer]
                time_similarity = abs(time_windows[seed][0] - time_windows[customer][0])
                combined = dist_similarity + 0.1 * time_similarity
                relatedness.append((combined, customer))

        relatedness.sort()

        for _, customer in relatedness[:min(num_remove-1, len(relatedness))]:
            to_remove.add(customer)

        # Remove from routes
        new_routes = []
        depot = routes[0][0]
        for route in routes:
            new_route = [depot]
            for customer in route[1:-1]:
                if customer not in to_remove:
                    new_route.append(customer)
            new_route.append(depot)
            if len(new_route) > 2:
                new_routes.append(new_route)

        return new_routes, list(to_remove)

    def greedy_insertion_tw(routes, removed_customers, depot):
        """Reinsert customers with time window checking"""
        uninserted = removed_customers.copy()

        def check_insertion_feasibility(route, customer, position):
            """Check if inserting customer at position is feasible"""
            new_route = route[:position] + [customer] + route[position:]

            # Check capacity
            load = sum(demands[c] for c in new_route[1:-1])
            if load > vehicle_capacity:
                return False

            # Check time windows
            current_time = time_windows[new_route[0]][0]
            for i in range(len(new_route) - 1):
                current_time += time_matrix[new_route[i]][new_route[i+1]]
                if current_time > time_windows[new_route[i+1]][1]:
                    return False
                current_time = max(current_time, time_windows[new_route[i+1]][0])
                current_time += service_times[new_route[i+1]]

            return True

        while uninserted:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost = float('inf')

            for customer in uninserted:
                # Try existing routes
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        if check_insertion_feasibility(route, customer, pos):
                            cost = (dist_matrix[route[pos-1]][customer] +
                                  dist_matrix[customer][route[pos]] -
                                  dist_matrix[route[pos-1]][route[pos]])

                            if cost < best_cost:
                                best_cost = cost
                                best_customer = customer
                                best_route_idx = r_idx
                                best_position = pos

            if best_customer is None:
                # Create new route
                routes.append([depot, uninserted.pop(0), depot])
            else:
                routes[best_route_idx].insert(best_position, best_customer)
                uninserted.remove(best_customer)

        return routes

    # ALNS main loop
    current_routes = copy.deepcopy(initial_solution)
    current_cost = calculate_cost(current_routes)

    best_routes = copy.deepcopy(current_routes)
    best_cost = current_cost

    depot = current_routes[0][0]

    # Count customers
    all_customers = []
    for route in current_routes:
        all_customers.extend(route[1:-1])
    num_remove = max(1, int(len(all_customers) * destroy_size))

    for iteration in range(iterations):
        # Destroy
        partial_routes, removed = shaw_removal_tw(current_routes, num_remove)

        # Repair
        new_routes = greedy_insertion_tw(partial_routes, removed, depot)

        # Evaluate
        new_cost = calculate_cost(new_routes)

        # Simulated annealing acceptance
        temperature = 100 * (1 - iteration / iterations)
        accept = (new_cost < current_cost or
                 random.random() < np.exp(-(new_cost - current_cost) / max(temperature, 1)))

        if accept:
            current_routes = new_routes
            current_cost = new_cost

            if new_cost < best_cost:
                best_routes = copy.deepcopy(new_routes)
                best_cost = new_cost

    return {
        'routes': best_routes,
        'total_distance': best_cost,
        'num_vehicles': len(best_routes)
    }
```

---

## Using OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_vrptw_ortools(dist_matrix, time_matrix, demands, time_windows,
                       service_times, vehicle_capacities, num_vehicles,
                       depot=0, time_limit=60):
    """
    Solve VRPTW using Google OR-Tools

    Most practical approach for real-world VRPTW

    Args:
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times at each location
        vehicle_capacities: vehicle capacities
        num_vehicles: number of vehicles
        depot: depot index
        time_limit: time limit in seconds

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)

    # Handle uniform fleet
    if isinstance(vehicle_capacities, (int, float)):
        vehicle_capacities = [vehicle_capacities] * num_vehicles

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)

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
        return int(time_matrix[from_node][to_node] + service_times[from_node])

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time window constraints
    routing.AddDimension(
        time_callback_index,
        30,  # allow waiting time
        3000,  # maximum time per vehicle
        False,  # don't force start cumul to zero
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')

    # Add time window constraints for each location
    for location_idx in range(n):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(
            int(time_windows[location_idx][0]),
            int(time_windows[location_idx][1])
        )

    # Add time window constraints for depot (all vehicles)
    for vehicle_id in range(num_vehicles):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        time_dimension.CumulVar(start_index).SetRange(
            int(time_windows[depot][0]),
            int(time_windows[depot][1])
        )
        time_dimension.CumulVar(end_index).SetRange(
            int(time_windows[depot][0]),
            int(time_windows[depot][1])
        )

    # Instantiate route start and end times to produce feasible times
    for i in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
        True,  # start cumul to zero
        'Capacity')

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = True

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        arrival_times = []
        total_distance = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            times = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                route.append(node)
                times.append(solution.Value(time_var))
                index = solution.Value(routing.NextVar(index))

            # Add depot at end
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route.append(node)
            times.append(solution.Value(time_var))

            if len(route) > 2:
                routes.append(route)
                arrival_times.append(times)

                # Calculate route distance
                route_distance = sum(dist_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1))
                total_distance += route_distance

        return {
            'status': 'Optimal' if solution.ObjectiveValue() > 0 else 'Feasible',
            'routes': routes,
            'arrival_times': arrival_times,
            'total_distance': total_distance,
            'num_vehicles_used': len(routes),
            'objective_value': solution.ObjectiveValue() / 100.0
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Example with visualization
def visualize_vrptw_solution(coordinates, routes, arrival_times, time_windows,
                            save_path=None):
    """
    Visualize VRPTW solution with Gantt chart

    Args:
        coordinates: list of (x, y) coordinates
        routes: list of routes
        arrival_times: arrival times at each location
        time_windows: time windows
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot routes
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for idx, route in enumerate(routes):
        route_coords = [coordinates[i] for i in route]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        ax1.plot(xs, ys, 'o-', color=colors[idx],
                linewidth=2, markersize=8,
                label=f'Vehicle {idx+1}')

    depot = coordinates[0]
    ax1.plot(depot[0], depot[1], 's', color='red',
            markersize=15, label='Depot', zorder=10)

    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('VRPTW Routes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Gantt chart
    for idx, (route, times) in enumerate(zip(routes, arrival_times)):
        for i in range(1, len(route) - 1):
            customer = route[i]
            arrival = times[i]
            tw_early, tw_late = time_windows[customer]

            # Draw time window
            ax2.add_patch(Rectangle((tw_early, idx - 0.3),
                                   tw_late - tw_early, 0.6,
                                   facecolor='lightgray', edgecolor='black',
                                   alpha=0.5))

            # Draw arrival/service
            ax2.plot(arrival, idx, 'o', color=colors[idx], markersize=10)
            ax2.text(arrival, idx + 0.4, f'C{customer}',
                    ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Vehicle')
    ax2.set_title('Schedule (Gantt Chart)')
    ax2.set_yticks(range(len(routes)))
    ax2.set_yticklabels([f'Vehicle {i+1}' for i in range(len(routes))])
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Complete example
if __name__ == "__main__":
    # Generate Solomon-like problem instance
    np.random.seed(42)

    n = 26  # 1 depot + 25 customers
    coordinates = np.random.rand(n, 2) * 100

    # Distance and time matrices
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            dist_matrix[i][j] = dist
            time_matrix[i][j] = dist / 40 * 60  # 40 km/h in minutes

    # Generate time windows (clustered)
    time_windows = [(0, 480)]  # Depot: 8-hour day

    for i in range(1, n):
        # Random center time
        center = random.randint(60, 420)
        width = random.randint(30, 90)
        time_windows.append((center - width, center + width))

    # Service times
    service_times = [0] + [random.randint(10, 20) for _ in range(n-1)]

    # Demands
    demands = [0] + [random.randint(5, 20) for _ in range(n-1)]

    vehicle_capacity = 100
    num_vehicles = 5

    print("Solving VRPTW with OR-Tools...")
    result = solve_vrptw_ortools(
        dist_matrix, time_matrix, demands, time_windows,
        service_times, vehicle_capacity, num_vehicles,
        time_limit=60
    )

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Vehicles Used: {result['num_vehicles_used']}/{num_vehicles}")

    print("\nRoute Details:")
    for i, (route, times) in enumerate(zip(result['routes'],
                                           result['arrival_times'])):
        print(f"\nVehicle {i+1}: {route}")
        print("  Arrival times:")
        for j, (customer, time) in enumerate(zip(route, times)):
            tw = time_windows[customer]
            print(f"    Stop {j}: Customer {customer} at time {time:.1f} "
                  f"(window: [{tw[0]:.1f}, {tw[1]:.1f}])")

    # Visualize
    visualize_vrptw_solution(coordinates, result['routes'],
                            result['arrival_times'], time_windows)
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **OR-Tools (Google)**: Best for practical VRPTW (highly recommended)
- **PuLP**: MIP modeling
- **VRPy**: Specialized VRP library with time windows support
- **python-mip**: Alternative MIP library

**Route Planning:**
- **OSRM**: Open Source Routing Machine (real-world distances)
- **Google Maps API**: Distance and time matrices
- **Valhalla**: Open-source routing engine

**Visualization:**
- **matplotlib**: Basic visualization
- **plotly**: Interactive Gantt charts
- **folium**: Map-based visualization

### Benchmark Instances

- **Solomon Instances**: Standard VRPTW benchmarks (R1, C1, RC1, R2, C2, RC2)
- **Gehring & Homberger**: Large-scale VRPTW instances
- **Sintef**: Comprehensive VRP test sets

---

## Common Challenges & Solutions

### Challenge: Tight Time Windows

**Problem:**
- Narrow time windows make problem highly constrained
- Many infeasible combinations

**Solutions:**
- Use time-oriented insertion criteria (Solomon's I1)
- Consider soft time windows with penalties
- Sequence-first, route-second approach
- Allow longer routes if necessary

### Challenge: Wide Time Horizon

**Problem:**
- Time windows span long periods (e.g., all-day delivery)
- Less constraint, but many possible solutions

**Solutions:**
- Use distance/cost as primary criterion
- Consider clustering customers by time preference
- Balance route duration

### Challenge: Mixed Hard/Soft Windows

**Problem:**
- Some customers have strict appointments
- Others are flexible

**Solutions:**
- Two-stage approach: satisfy hard windows first
- Use penalty costs for soft window violations
- Prioritize hard window customers in insertion

### Challenge: Real-World Timing

**Problem:**
- Traffic varies by time of day
- Stochastic travel times

**Solutions:**
- Time-dependent travel time matrices
- Add time buffers/slack
- Stochastic VRPTW models
- Real-time re-optimization

---

## Output Format

### VRPTW Solution Report

**Problem Instance:**
- Customers: 50
- Time Horizon: 8:00 AM - 6:00 PM (600 minutes)
- Vehicles: 5 (capacity: 100 units each)
- Average time window width: 60 minutes

**Solution Quality:**

| Metric | Value |
|--------|-------|
| Total Distance | 892.4 km |
| Total Time | 18.2 hours |
| Vehicles Used | 5 / 5 |
| Average Route Duration | 3.64 hours |
| Time Window Violations | 0 (feasible) |
| Average Waiting Time | 12.3 minutes |

**Route Schedule:**

**Vehicle 1:** Start 8:00 AM

| Stop | Customer | Arrival | Time Window | Wait | Service | Depart |
|------|----------|---------|-------------|------|---------|--------|
| 0 | Depot | 8:00 | [8:00-18:00] | 0 | 0 | 8:00 |
| 1 | C12 | 8:23 | [8:15-9:15] | 0 | 15 | 8:38 |
| 2 | C7 | 9:05 | [9:00-10:00] | 0 | 12 | 9:17 |
| 3 | C34 | 9:42 | [9:30-10:30] | 0 | 10 | 9:52 |
| 4 | Depot | 10:35 | [8:00-18:00] | - | - | - |

**Route Distance:** 187.2 km
**Route Duration:** 2:35
**Total Load:** 97 / 100 units

[Continue for all vehicles...]

**Statistics:**
- On-time arrivals: 100%
- Early arrivals requiring wait: 8%
- Average slack in time windows: 23 minutes

---

## Questions to Ask

If you need more context:
1. Do all customers have time windows or only a subset?
2. Are time windows hard (must satisfy) or soft (can penalize violations)?
3. How wide are the time windows typically?
4. What's the planning horizon? (8 hours, 12 hours, 24 hours)
5. Do you have actual travel times or should we estimate from distances?
6. Are service times significant?
7. Is minimizing vehicles or total distance the primary goal?
8. Do routes have maximum duration limits?
9. Are there driver break requirements?
10. Is this recurring daily or a one-time problem?

---

## Related Skills

- **vehicle-routing-problem**: For basic VRP without time windows
- **traveling-salesman-problem**: For single vehicle routing
- **pickup-delivery-problem**: For paired pickup-delivery with time windows
- **capacitated-vrp**: For pure capacity-focused VRP
- **multi-depot-vrp**: For multiple depot routing
- **route-optimization**: For practical routing applications
- **last-mile-delivery**: For final delivery optimization
- **fleet-management**: For fleet operations

---
name: warehouse-automation
description: When the user wants to implement warehouse automation, evaluate automation technologies, or design automated material handling systems. Also use when the user mentions "warehouse robotics," "automated storage," "AS/RS," "goods-to-person," "conveyor systems," "sortation," "AMR," "AGV," "automated picking," or "warehouse automation ROI." For warehouse layout design, see warehouse-design. For order fulfillment, see order-fulfillment.
---

# Warehouse Automation

You are an expert in warehouse automation and material handling systems. Your goal is to help evaluate, design, and implement automation technologies that improve operational efficiency, reduce labor costs, increase accuracy, and enable scalability.

## Initial Assessment

Before implementing automation, understand:

1. **Business Drivers**
   - What's driving automation interest? (labor, accuracy, capacity, throughput)
   - Labor challenges? (availability, cost, turnover)
   - Growth projections? (volume, SKUs, expansion)
   - Service requirements? (speed, accuracy, scalability)

2. **Current Operations**
   - Order volume? (current and projected)
   - Order profile? (lines/order, units/order, size)
   - Current processes and pain points?
   - Warehouse size and configuration?
   - Current technology? (WMS, conveyors, equipment)

3. **Financial Context**
   - Capital budget available?
   - ROI expectations and timeframe?
   - Labor costs (fully loaded with benefits)?
   - Current operational costs?

4. **Constraints**
   - Facility constraints? (ceiling height, floor load, space)
   - Existing systems to integrate with?
   - Operational constraints? (24/7, seasonal peaks)
   - Regulatory requirements? (FDA, safety)

---

## Warehouse Automation Framework

### Automation Technology Spectrum

**Level 0: Manual Operations**
- Paper pick lists
- Manual processes
- High labor, low capital

**Level 1: Basic Technology**
- WMS and RF scanning
- Barcode tracking
- Pick-to-light (simple)
- Basic conveyors

**Level 2: Semi-Automated**
- Conveyor systems
- Sortation systems
- Vertical lift modules (VLM)
- Carousels
- Voice picking

**Level 3: Highly Automated**
- AS/RS (automated storage/retrieval)
- Goods-to-person systems
- AMRs (autonomous mobile robots)
- Automated picking (robotic arms)
- AutoStore / Cube storage

**Level 4: Fully Automated (Lights-Out)**
- End-to-end automation
- Minimal human intervention
- AI-driven optimization
- Self-optimizing systems

---

## Automation Technologies

### 1. Conveyor & Sortation Systems

**Use Cases:**
- Move products between zones
- Sort orders to packing stations or shipping lanes
- Buffer and accumulate inventory

**Types:**

| Type | Throughput | Cost | Best For |
|------|-----------|------|----------|
| Belt Conveyors | Medium | $ | General transport |
| Roller Conveyors | Medium | $ | Pallets, cartons |
| Sliding Shoe Sorter | 5K-15K units/hr | $$$ | High-speed parcel sorting |
| Cross-Belt Sorter | 10K-30K+ units/hr | $$$$ | Very high throughput, fragile items |
| Tilt-Tray Sorter | 5K-20K units/hr | $$$ | Medium to high speed |
| Bomb-Bay Sorter | 3K-10K units/hr | $$ | Case sorting |

```python
import numpy as np
import pandas as pd

def conveyor_sortation_sizing(orders_per_day, items_per_order=5,
                              operating_hours=16, target_utilization=0.75,
                              sorter_type='sliding_shoe'):
    """
    Size conveyor and sortation system

    Parameters:
    - orders_per_day: Daily order volume
    - items_per_order: Average items per order
    - operating_hours: Hours of operation per day
    - target_utilization: Target system utilization (0.75 = 75%)
    - sorter_type: Type of sorter

    Returns:
    - System specifications and cost estimate
    """

    # Calculate required throughput
    items_per_day = orders_per_day * items_per_order
    items_per_hour = items_per_day / operating_hours

    # Required capacity (accounting for utilization)
    required_capacity = items_per_hour / target_utilization

    # Sorter specifications and costs
    sorter_specs = {
        'sliding_shoe': {
            'capacity_range': (5000, 15000),
            'cost_range': (1_500_000, 3_000_000),
            'cost_per_linear_ft': 800
        },
        'cross_belt': {
            'capacity_range': (10000, 30000),
            'cost_range': (3_000_000, 8_000_000),
            'cost_per_linear_ft': 1500
        },
        'tilt_tray': {
            'capacity_range': (5000, 20000),
            'cost_range': (2_000_000, 5_000_000),
            'cost_per_linear_ft': 1000
        }
    }

    spec = sorter_specs[sorter_type]

    # Check if within capacity range
    if required_capacity < spec['capacity_range'][0]:
        recommendation = f"Over-specified. Consider lower throughput solution or {sorter_type} is fine."
    elif required_capacity > spec['capacity_range'][1]:
        recommendation = "Under-specified. Need higher capacity sorter or multiple systems."
    else:
        recommendation = f"{sorter_type.replace('_', ' ').title()} is appropriate."

    # Estimate cost (assume mid-range)
    estimated_cost = np.mean(spec['cost_range'])

    # Estimate conveyor length needed (assume 300 ft)
    conveyor_length_ft = 300
    conveyor_cost = conveyor_length_ft * 500  # $500/ft for belt conveyor

    total_cost = estimated_cost + conveyor_cost

    return {
        'items_per_hour': round(items_per_hour, 0),
        'required_capacity': round(required_capacity, 0),
        'sorter_type': sorter_type,
        'recommendation': recommendation,
        'sorter_cost': estimated_cost,
        'conveyor_cost': conveyor_cost,
        'total_cost': total_cost,
        'cost_per_order': round(total_cost / (orders_per_day * 250), 2)  # Amortized over 250 days/year
    }

# Example
sortation = conveyor_sortation_sizing(
    orders_per_day=5000,
    items_per_order=6,
    operating_hours=16,
    sorter_type='sliding_shoe'
)

print(f"Required capacity: {sortation['required_capacity']} items/hour")
print(f"Recommendation: {sortation['recommendation']}")
print(f"Total cost: ${sortation['total_cost']:,.0f}")
print(f"Cost per order (amortized): ${sortation['cost_per_order']}")
```

### 2. Automated Storage & Retrieval Systems (AS/RS)

**Use Cases:**
- Dense storage (maximize cube utilization)
- High throughput put-away and retrieval
- Accurate inventory management
- Reduce travel time

**Types:**

**Unit-Load AS/RS:**
- Full pallet storage
- 40+ ft high
- 100-200 cycles/hour
- Cost: $2M-$10M+

**Mini-Load AS/RS:**
- Tote/carton storage
- 20-40 ft high
- 200-400 cycles/hour
- Cost: $1M-$5M

**Shuttle Systems:**
- High-density storage
- Scalable (add shuttles as needed)
- 400-800+ cycles/hour
- Cost: $2M-$8M

```python
def asrs_capacity_analysis(pallet_positions_needed, storage_height_ft,
                          throughput_pallets_per_hour, system_type='unit_load'):
    """
    Analyze AS/RS capacity and cost

    Parameters:
    - pallet_positions_needed: Total storage positions required
    - storage_height_ft: Available storage height
    - throughput_pallets_per_hour: Required throughput (in/out)
    - system_type: 'unit_load', 'mini_load', 'shuttle'

    Returns:
    - System configuration and cost
    """

    system_specs = {
        'unit_load': {
            'positions_per_aisle': 1000,  # Typical
            'cycles_per_hour': 150,
            'cost_per_aisle': 1_500_000,
            'height_min': 40,
            'footprint_per_aisle_sqft': 2000
        },
        'mini_load': {
            'positions_per_aisle': 1500,
            'cycles_per_hour': 300,
            'cost_per_aisle': 800_000,
            'height_min': 20,
            'footprint_per_aisle_sqft': 1500
        },
        'shuttle': {
            'positions_per_aisle': 2000,
            'cycles_per_hour': 600,
            'cost_per_shuttle': 200_000,
            'cost_base_structure': 2_000_000,
            'height_min': 20,
            'footprint_per_aisle_sqft': 1200
        }
    }

    spec = system_specs[system_type]

    # Check height requirement
    if storage_height_ft < spec['height_min']:
        return {
            'error': f"{system_type} requires minimum {spec['height_min']} ft clear height. "
                    f"Only {storage_height_ft} ft available."
        }

    # Calculate aisles needed (based on storage)
    aisles_needed_storage = np.ceil(pallet_positions_needed / spec['positions_per_aisle'])

    # Calculate aisles needed (based on throughput)
    if system_type == 'shuttle':
        # Shuttles can be added independently
        shuttles_needed = np.ceil(throughput_pallets_per_hour / spec['cycles_per_hour'])
        cost = spec['cost_base_structure'] + (shuttles_needed * spec['cost_per_shuttle'])
        aisles_needed_throughput = 1  # Flexible with shuttles
    else:
        # Cranes are per aisle
        aisles_needed_throughput = np.ceil(throughput_pallets_per_hour / spec['cycles_per_hour'])
        aisles_needed = max(aisles_needed_storage, aisles_needed_throughput)
        cost = aisles_needed * spec['cost_per_aisle']

    if system_type != 'shuttle':
        aisles_needed = max(aisles_needed_storage, aisles_needed_throughput)
    else:
        aisles_needed = aisles_needed_storage

    # Footprint
    total_footprint = aisles_needed * spec['footprint_per_aisle_sqft']

    # Capacity per aisle/shuttle
    if system_type == 'shuttle':
        total_positions = aisles_needed * spec['positions_per_aisle']
        total_throughput = shuttles_needed * spec['cycles_per_hour']
    else:
        total_positions = aisles_needed * spec['positions_per_aisle']
        total_throughput = aisles_needed * spec['cycles_per_hour']

    return {
        'system_type': system_type,
        'aisles_or_shuttles': int(aisles_needed if system_type != 'shuttle' else shuttles_needed),
        'total_positions': int(total_positions),
        'total_throughput_per_hour': int(total_throughput),
        'footprint_sqft': int(total_footprint),
        'total_cost': int(cost),
        'cost_per_position': round(cost / total_positions, 2),
        'utilization_%': round(min(
            pallet_positions_needed / total_positions,
            throughput_pallets_per_hour / total_throughput
        ) * 100, 1)
    }

# Example
asrs = asrs_capacity_analysis(
    pallet_positions_needed=5000,
    storage_height_ft=45,
    throughput_pallets_per_hour=300,
    system_type='unit_load'
)

print(f"System: {asrs['system_type']}")
print(f"Aisles needed: {asrs['aisles_or_shuttles']}")
print(f"Total positions: {asrs['total_positions']}")
print(f"Total cost: ${asrs['total_cost']:,.0f}")
print(f"Cost per position: ${asrs['cost_per_position']}")
```

### 3. Goods-to-Person (GTP) Systems

**Description:**
- Bring inventory to picker (vs. picker traveling)
- Dramatically reduces travel time
- High pick rates (200-400+ lines/hour/person)

**Types:**

**Vertical Lift Modules (VLM):**
- Vertical storage with elevator
- 30-50 ft high
- Good for small parts
- Cost: $150K-$300K per unit

**Horizontal Carousels:**
- Rotating shelves
- Picker waits, carousel rotates to item
- Cost: $80K-$150K per unit

**Vertical Carousels:**
- Similar to VLM but rotating
- Cost: $100K-$200K per unit

**Shuttle-Based GTP (AutoStore, Exotec, etc.):**
- Cube storage with robots retrieving bins
- Very high density
- Scalable
- Cost: $2M-$10M+

```python
def gtp_system_comparison(pick_lines_per_day, operating_hours=16,
                         warehouse_sqft=50000):
    """
    Compare goods-to-person system options

    Parameters:
    - pick_lines_per_day: Daily pick volume (lines)
    - operating_hours: Operating hours per day
    - warehouse_sqft: Available warehouse space

    Returns:
    - Comparison of GTP options
    """

    pick_lines_per_hour = pick_lines_per_day / operating_hours

    systems = {
        'Vertical Lift Module (VLM)': {
            'pick_rate_per_hour': 150,
            'cost_per_unit': 200_000,
            'footprint_per_unit_sqft': 150,
            'max_height_ft': 50,
            'scalability': 'Modular (add units)',
            'best_for': 'Small parts, medium volume'
        },
        'Horizontal Carousel': {
            'pick_rate_per_hour': 120,
            'cost_per_unit': 100_000,
            'footprint_per_unit_sqft': 300,
            'max_height_ft': 10,
            'scalability': 'Limited (fixed size)',
            'best_for': 'Small parts, lower volume'
        },
        'Shuttle GTP (AutoStore-like)': {
            'pick_rate_per_hour': 250,
            'cost_per_unit': 5_000_000,  # System cost
            'footprint_per_unit_sqft': 10000,  # System footprint
            'max_height_ft': 20,
            'scalability': 'Highly scalable (add robots)',
            'best_for': 'High volume, e-commerce'
        }
    }

    recommendations = []

    for system_name, specs in systems.items():
        # Calculate units needed
        if 'AutoStore' in system_name:
            # One system, calculate ports needed
            ports_needed = np.ceil(pick_lines_per_hour / specs['pick_rate_per_hour'])
            units_needed = 1
            total_cost = specs['cost_per_unit']
            footprint = specs['footprint_per_unit_sqft']
        else:
            units_needed = np.ceil(pick_lines_per_hour / specs['pick_rate_per_hour'])
            total_cost = units_needed * specs['cost_per_unit']
            footprint = units_needed * specs['footprint_per_unit_sqft']

        # Check if fits in warehouse
        fits = footprint < warehouse_sqft * 0.5  # Use max 50% of space

        recommendations.append({
            'system': system_name,
            'units_or_ports': int(units_needed if 'AutoStore' not in system_name else ports_needed),
            'total_cost': int(total_cost),
            'footprint_sqft': int(footprint),
            'fits_in_warehouse': fits,
            'pick_rate_total': int(units_needed * specs['pick_rate_per_hour']
                                  if 'AutoStore' not in system_name
                                  else ports_needed * specs['pick_rate_per_hour']),
            'best_for': specs['best_for']
        })

    return pd.DataFrame(recommendations)

# Example
gtp_comparison = gtp_system_comparison(
    pick_lines_per_day=20000,
    operating_hours=16,
    warehouse_sqft=50000
)

print("Goods-to-Person System Comparison:")
print(gtp_comparison)
```

### 4. Autonomous Mobile Robots (AMR) & AGVs

**AMRs (Autonomous Mobile Robots):**
- Navigate autonomously (no fixed paths)
- Flexible and scalable
- Examples: Locus Robotics, 6 River Systems, GreyOrange
- Cost: $30K-$50K per robot

**AGVs (Automated Guided Vehicles):**
- Follow fixed paths (wires, magnets, or tape)
- Less flexible but robust
- Cost: $50K-$150K per vehicle

**Use Cases:**
- Picking assistance (collaborative picking)
- Inventory transport (putaway, replenishment)
- Sortation and consolidation

```python
def amr_fleet_sizing(orders_per_day, lines_per_order=5, avg_travel_distance_ft=500,
                    robot_speed_ft_per_min=150, pick_time_per_line_sec=10,
                    operating_hours=16):
    """
    Calculate AMR fleet size requirements

    Parameters:
    - orders_per_day: Daily order volume
    - lines_per_order: Average lines per order
    - avg_travel_distance_ft: Average travel per order
    - robot_speed_ft_per_min: Robot travel speed
    - pick_time_per_line_sec: Time to pick each line
    - operating_hours: Operating hours per day

    Returns:
    - AMR fleet requirements and cost
    """

    # Calculate total picks
    picks_per_day = orders_per_day * lines_per_order
    picks_per_hour = picks_per_day / operating_hours

    # Time per order
    travel_time_min = avg_travel_distance_ft / robot_speed_ft_per_min
    pick_time_min = (lines_per_order * pick_time_per_sec) / 60
    total_time_per_order_min = travel_time_min + pick_time_min

    # Orders per robot per hour
    orders_per_robot_per_hour = 60 / total_time_per_order_min

    # Robots needed
    robots_needed_raw = (orders_per_day / operating_hours) / orders_per_robot_per_hour

    # Add buffer for charging, maintenance (20%)
    robots_needed = np.ceil(robots_needed_raw * 1.2)

    # Cost
    cost_per_robot = 40000  # Average AMR cost
    fleet_cost = robots_needed * cost_per_robot

    # Annual operating cost (maintenance, support)
    annual_operating_cost = fleet_cost * 0.15  # 15% of capital

    # Productivity improvement
    # Traditional picking: 100 lines/hour
    # AMR-assisted: 150 lines/hour (50% improvement)
    traditional_pickers = picks_per_day / (100 * operating_hours)
    amr_pickers = picks_per_day / (150 * operating_hours)
    labor_reduction = traditional_pickers - amr_pickers

    return {
        'picks_per_day': picks_per_day,
        'picks_per_hour': round(picks_per_hour, 0),
        'orders_per_robot_per_hour': round(orders_per_robot_per_hour, 1),
        'robots_needed': int(robots_needed),
        'fleet_cost': int(fleet_cost),
        'annual_operating_cost': int(annual_operating_cost),
        'traditional_pickers_needed': round(traditional_pickers, 1),
        'amr_assisted_pickers_needed': round(amr_pickers, 1),
        'labor_reduction': round(labor_reduction, 1),
        'labor_reduction_%': round((labor_reduction / traditional_pickers) * 100, 1)
    }

# Example
amr_fleet = amr_fleet_sizing(
    orders_per_day=2000,
    lines_per_order=5,
    avg_travel_distance_ft=500,
    operating_hours=16
)

print(f"AMRs needed: {amr_fleet['robots_needed']}")
print(f"Fleet cost: ${amr_fleet['fleet_cost']:,.0f}")
print(f"Labor reduction: {amr_fleet['labor_reduction']} pickers ({amr_fleet['labor_reduction_%']}%)")
```

### 5. Automated Picking Systems

**Robotic Piece Picking:**
- Robotic arms with grippers/suction
- Pick individual items
- Technology still maturing
- Examples: RightHand Robotics, Berkshire Grey
- Cost: $250K-$500K per cell

**Pick-to-Light:**
- Lights guide pickers
- Simple, effective
- Cost: $200-$500 per light position

**Voice Picking:**
- Hands-free, eyes-free
- Headset directs picker
- Cost: $2K-$3K per headset + software

```python
def picking_technology_comparison(pick_lines_per_day, accuracy_target=0.995):
    """
    Compare picking technology options

    Parameters:
    - pick_lines_per_day: Daily pick volume
    - accuracy_target: Target pick accuracy (e.g., 0.995 = 99.5%)

    Returns:
    - Technology comparison
    """

    technologies = {
        'Manual (Paper)': {
            'pick_rate': 80,
            'accuracy': 0.97,
            'cost_per_picker': 500,  # RF scanner only
            'training_time_hours': 8
        },
        'RF Scanning': {
            'pick_rate': 100,
            'accuracy': 0.99,
            'cost_per_picker': 3000,  # RF device
            'training_time_hours': 16
        },
        'Voice Picking': {
            'pick_rate': 120,
            'accuracy': 0.995,
            'cost_per_picker': 5000,  # Headset + software
            'training_time_hours': 24
        },
        'Pick-to-Light': {
            'pick_rate': 150,
            'accuracy': 0.998,
            'cost_per_picker': 25000,  # Light systems
            'training_time_hours': 8
        },
        'AMR-Assisted': {
            'pick_rate': 150,
            'accuracy': 0.998,
            'cost_per_picker': 40000,  # Robot amortized
            'training_time_hours': 16
        },
        'Robotic Picking': {
            'pick_rate': 200,
            'accuracy': 0.999,
            'cost_per_picker': 300000,  # Robotic cell
            'training_time_hours': 40
        }
    }

    comparison = []

    for tech_name, specs in technologies.items():
        pickers_needed = np.ceil(pick_lines_per_day / (specs['pick_rate'] * 8))  # 8 hr shifts
        total_cost = pickers_needed * specs['cost_per_picker']
        meets_accuracy = specs['accuracy'] >= accuracy_target

        comparison.append({
            'technology': tech_name,
            'pick_rate': specs['pick_rate'],
            'accuracy_%': specs['accuracy'] * 100,
            'pickers_needed': int(pickers_needed),
            'total_cost': int(total_cost),
            'meets_accuracy_target': meets_accuracy,
            'training_hours': specs['training_time_hours']
        })

    return pd.DataFrame(comparison)

# Example
picking_comparison = picking_technology_comparison(
    pick_lines_per_day=10000,
    accuracy_target=0.995
)

print("Picking Technology Comparison:")
print(picking_comparison)
```

---

## Automation ROI Analysis

### ROI Calculation Framework

```python
def automation_roi_analysis(capital_cost, current_labor_cost_annual,
                           labor_reduction_pct, productivity_improvement_pct=0,
                           accuracy_improvement_savings=0,
                           maintenance_cost_pct=0.15,
                           useful_life_years=7, discount_rate=0.10):
    """
    Calculate comprehensive ROI for warehouse automation

    Parameters:
    - capital_cost: Upfront automation investment
    - current_labor_cost_annual: Current annual labor cost
    - labor_reduction_pct: Labor reduction (0.40 = 40%)
    - productivity_improvement_pct: Throughput increase for same labor
    - accuracy_improvement_savings: Annual savings from improved accuracy
    - maintenance_cost_pct: Annual maintenance as % of capital
    - useful_life_years: Expected system life
    - discount_rate: Discount rate for NPV

    Returns:
    - ROI metrics
    """

    # Annual benefits
    labor_savings = current_labor_cost_annual * labor_reduction_pct
    productivity_savings = current_labor_cost_annual * productivity_improvement_pct / (1 + productivity_improvement_pct)
    accuracy_savings = accuracy_improvement_savings

    total_annual_savings = labor_savings + productivity_savings + accuracy_savings

    # Annual costs
    maintenance_cost = capital_cost * maintenance_cost_pct
    annual_net_savings = total_annual_savings - maintenance_cost

    # Simple payback
    simple_payback_years = capital_cost / annual_net_savings if annual_net_savings > 0 else 999

    # NPV calculation
    npv = -capital_cost
    for year in range(1, useful_life_years + 1):
        npv += annual_net_savings / ((1 + discount_rate) ** year)

    # IRR approximation (simplified)
    roi_total = (annual_net_savings * useful_life_years - capital_cost) / capital_cost

    return {
        'capital_cost': capital_cost,
        'annual_labor_savings': round(labor_savings, 0),
        'annual_productivity_savings': round(productivity_savings, 0),
        'annual_accuracy_savings': round(accuracy_savings, 0),
        'total_annual_savings': round(total_annual_savings, 0),
        'annual_maintenance': round(maintenance_cost, 0),
        'annual_net_savings': round(annual_net_savings, 0),
        'simple_payback_years': round(simple_payback_years, 2),
        'npv': round(npv, 0),
        'roi_%_over_life': round(roi_total * 100, 1)
    }

# Example
roi = automation_roi_analysis(
    capital_cost=3_000_000,
    current_labor_cost_annual=2_500_000,
    labor_reduction_pct=0.40,
    productivity_improvement_pct=0.20,
    accuracy_improvement_savings=200_000,
    maintenance_cost_pct=0.15,
    useful_life_years=7
)

print("Automation ROI Analysis:")
for metric, value in roi.items():
    if isinstance(value, (int, float)) and value > 1000:
        print(f"  {metric}: ${value:,.0f}")
    else:
        print(f"  {metric}: {value}")
```

### Break-Even Analysis

```python
def automation_breakeven_analysis(capital_cost, annual_net_savings,
                                 labor_cost_inflation=0.03):
    """
    Calculate break-even timeline considering labor inflation

    Parameters:
    - capital_cost: Automation investment
    - annual_net_savings: First year net savings
    - labor_cost_inflation: Annual labor cost increase (3%)

    Returns:
    - Year-by-year break-even analysis
    """

    cumulative_savings = 0
    year = 0

    breakeven_table = []

    while cumulative_savings < capital_cost and year < 15:
        year += 1

        # Savings increase with labor inflation
        year_savings = annual_net_savings * ((1 + labor_cost_inflation) ** (year - 1))
        cumulative_savings += year_savings

        breakeven_table.append({
            'year': year,
            'annual_savings': round(year_savings, 0),
            'cumulative_savings': round(cumulative_savings, 0),
            'remaining_to_breakeven': round(max(0, capital_cost - cumulative_savings), 0),
            'roi_%': round((cumulative_savings / capital_cost - 1) * 100, 1)
        })

        if cumulative_savings >= capital_cost:
            break

    return pd.DataFrame(breakeven_table)

# Example
breakeven = automation_breakeven_analysis(
    capital_cost=3_000_000,
    annual_net_savings=700_000,
    labor_cost_inflation=0.03
)

print("\nBreak-Even Analysis:")
print(breakeven)
```

---

## Automation Selection Framework

### Decision Matrix

```python
def automation_selection_decision(volume_level, sku_count, order_profile,
                                 labor_availability, capital_budget):
    """
    Recommend automation technologies based on operational profile

    Parameters:
    - volume_level: 'low' (<1K orders/day), 'medium' (1K-5K), 'high' (>5K)
    - sku_count: 'low' (<1K), 'medium' (1K-10K), 'high' (>10K)
    - order_profile: 'pallet', 'case', 'each', 'mixed'
    - labor_availability: 'abundant', 'moderate', 'scarce'
    - capital_budget: 'low' (<$500K), 'medium' ($500K-$3M), 'high' (>$3M)

    Returns:
    - Recommended automation technologies
    """

    recommendations = []

    # High volume + capital = full automation
    if volume_level == 'high' and capital_budget == 'high':
        recommendations.append({
            'technology': 'Shuttle-based GTP (AutoStore, Exotec)',
            'priority': 1,
            'reason': 'High volume justifies high automation, excellent ROI',
            'est_cost': '$5M-$15M'
        })
        recommendations.append({
            'technology': 'Cross-Belt Sorter',
            'priority': 2,
            'reason': 'High throughput sortation needed',
            'est_cost': '$3M-$8M'
        })

    # Medium volume, e-commerce profile
    elif volume_level == 'medium' and order_profile == 'each':
        recommendations.append({
            'technology': 'AMR Fleet + Pick-to-Light',
            'priority': 1,
            'reason': 'Scalable, flexible for e-commerce picking',
            'est_cost': '$500K-$2M'
        })
        recommendations.append({
            'technology': 'Vertical Lift Modules (VLM)',
            'priority': 2,
            'reason': 'Goods-to-person for fast movers',
            'est_cost': '$400K-$1M'
        })

    # Pallet-focused operations
    elif order_profile == 'pallet':
        recommendations.append({
            'technology': 'AS/RS Unit-Load',
            'priority': 1,
            'reason': 'Dense pallet storage, high throughput',
            'est_cost': '$2M-$8M'
        })
        recommendations.append({
            'technology': 'AGV Pallet Movers',
            'priority': 2,
            'reason': 'Automate pallet transport',
            'est_cost': '$500K-$1.5M'
        })

    # Labor scarcity driver
    elif labor_availability == 'scarce':
        recommendations.append({
            'technology': 'AMR Fleet',
            'priority': 1,
            'reason': 'Reduce labor dependency, scalable',
            'est_cost': '$300K-$1M'
        })
        recommendations.append({
            'technology': 'Voice Picking',
            'priority': 2,
            'reason': 'Improve productivity of available labor',
            'est_cost': '$50K-$200K'
        })

    # Low budget
    elif capital_budget == 'low':
        recommendations.append({
            'technology': 'WMS + RF Scanning',
            'priority': 1,
            'reason': 'Foundation for efficiency, low cost',
            'est_cost': '$100K-$300K'
        })
        recommendations.append({
            'technology': 'Conveyor System (Basic)',
            'priority': 2,
            'reason': 'Improve flow at modest cost',
            'est_cost': '$100K-$500K'
        })

    else:
        # Default recommendations
        recommendations.append({
            'technology': 'Start with WMS + RF Scanning',
            'priority': 1,
            'reason': 'Build foundation before advanced automation',
            'est_cost': '$100K-$300K'
        })

    return pd.DataFrame(recommendations)

# Example
automation_recs = automation_selection_decision(
    volume_level='medium',
    sku_count='high',
    order_profile='each',
    labor_availability='scarce',
    capital_budget='medium'
)

print("Automation Recommendations:")
print(automation_recs)
```

---

## Implementation Best Practices

### Phased Implementation Approach

**Phase 1: Foundation (Months 1-6)**
- Implement/upgrade WMS
- RF scanning and barcode infrastructure
- Process optimization (slotting, wave planning)
- Data clean-up and validation

**Phase 2: Targeted Automation (Months 7-12)**
- Automate specific pain points
- Pilot technologies (small scale)
- Measure results and refine
- Build internal expertise

**Phase 3: Scale (Year 2)**
- Expand successful pilots
- Add capacity as volume grows
- Integration and optimization
- Advanced features

**Phase 4: Continuous Improvement (Ongoing)**
- Monitor performance
- Upgrade and enhance
- New technologies
- Process refinement

### Critical Success Factors

1. **Start with Good Processes**: Automate good processes, not bad ones
2. **Strong Project Management**: Complex integrations, long timelines
3. **Change Management**: Train staff, manage resistance
4. **Vendor Selection**: Proven technology, strong support
5. **Integration**: WMS, ERP, warehouse systems
6. **Testing**: Thorough UAT before go-live
7. **Scalability**: Plan for growth
8. **Flexibility**: Accommodate changing business needs

---

## Common Challenges & Solutions

### Challenge: High Capital Cost

**Problem:**
- $2M-$10M+ investment
- Long payback periods
- Risk of technology obsolescence

**Solutions:**
- Phase approach (start small, prove ROI)
- Consider leasing vs. buying
- Focus on high-ROI areas first
- Robotics-as-a-Service (RaaS) models
- 3PL partnership (let them automate)

### Challenge: Integration Complexity

**Problem:**
- Multiple systems to integrate (WMS, ERP, automation)
- Custom interfaces and middleware
- Testing and validation time-consuming

**Solutions:**
- Choose automation compatible with WMS
- Experienced systems integrator
- Phased cutover (not big bang)
- Extensive testing environment
- Buffer time in project plan

### Challenge: Labor Displacement

**Problem:**
- Workers fear job loss
- Resistance to change
- Union concerns

**Solutions:**
- Communicate vision (augment, not replace)
- Retrain workers for new roles (maintenance, monitoring)
- Natural attrition and redeployment
- Emphasize safety and ergonomic benefits
- Involve workers in design process

### Challenge: Throughput Not Meeting Expectations

**Problem:**
- System runs slower than promised
- Downtime higher than expected
- Process bottlenecks

**Solutions:**
- Realistic vendor expectations (contractual SLAs)
- Pilot/proof-of-concept before full rollout
- Proper preventive maintenance
- Redundancy in design (extra capacity)
- Continuous optimization (tuning)

### Challenge: Changing Business Requirements

**Problem:**
- Automation designed for today's needs
- Business changes (SKUs, volume, channels)
- Inflexible systems

**Solutions:**
- Design for flexibility (modular, scalable)
- AMRs vs. fixed automation (more flexible)
- Overcapacity buffer (30-50%)
- Regular technology refresh cycles
- Cloud-based systems (easier to update)

---

## Emerging Technologies

### Innovations to Watch

**1. AI-Powered Robotics**
- Machine learning for picking (handle any item)
- Vision systems for bin picking
- Adaptive grasping

**2. Collaborative Robots (Cobots)**
- Work alongside humans safely
- Flexible deployment
- Lower cost than full automation

**3. Drones**
- Inventory scanning and cycle counting
- Read barcodes at height
- Autonomous flight

**4. Exoskeletons**
- Wearable assistance for workers
- Reduce fatigue and injury
- Augment human capability

**5. Digital Twins**
- Virtual warehouse simulation
- Test automation before building
- Ongoing optimization

**6. 5G Connectivity**
- Enable more real-time automation
- Support dense robot fleets
- IoT sensor networks

---

## Tools & Software

### Warehouse Control Systems (WCS)

**Purpose:**
- Orchestrate automation equipment
- Interface between WMS and automation
- Real-time control and optimization

**Major Vendors:**
- Dematic
- Honeywell Intelligrated
- Vanderlande
- TGW Logistics
- Swisslog

### Simulation Software

**Purpose:**
- Model warehouse operations
- Test automation scenarios
- Optimize layouts and throughput

**Tools:**
- FlexSim
- AnyLogic
- Simio
- Arena
- AutoMod

---

## Output Format

### Automation Feasibility Study

**Executive Summary:**
- Automation recommendation
- Expected ROI and payback
- Capital investment required
- Implementation timeline

**Current State Analysis:**

| Metric | Current | Pain Points |
|--------|---------|-------------|
| Orders/Day | 3,000 | Peaks to 8,000, can't scale |
| Labor Cost/Year | $3.5M | High turnover, tight market |
| Pick Rate | 85 lines/hr | Low productivity |
| Order Accuracy | 97.5% | Below target 99% |
| Space Utilization | 65% | Poor cube utilization |

**Recommended Automation:**

| Technology | Purpose | Capacity | Cost | ROI |
|-----------|---------|----------|------|-----|
| AMR Fleet (20 robots) | Picking assistance | +40% productivity | $800K | 2.1 yrs |
| Shuttle GTP System | Dense storage, GTP | 5K positions | $4M | 3.5 yrs |
| Sliding Shoe Sorter | Order sortation | 10K units/hr | $2M | 2.8 yrs |
| **Total** | - | - | **$6.8M** | **3.1 yrs** |

**Financial Analysis:**

| Metric | Value |
|--------|-------|
| Total Capital Investment | $6.8M |
| Annual Labor Savings | $1.4M (40% reduction) |
| Annual Productivity Gain | $500K (50% more throughput) |
| Annual Accuracy Savings | $150K (reduced errors) |
| Annual Net Savings | $1.65M (after $400K maintenance) |
| Simple Payback | 4.1 years |
| 7-Year NPV (10% discount) | $2.1M |
| ROI (7 years) | 75% |

**Implementation Plan:**
- **Months 1-3**: Design, engineering, vendor selection
- **Months 4-9**: Equipment fabrication and delivery
- **Months 10-12**: Installation and integration
- **Months 13-14**: Testing and training
- **Month 15**: Go-live
- **Months 16-18**: Ramp-up and optimization

**Risk Mitigation:**
- Pilot AMRs before full fleet purchase
- Phased cutover (one zone at a time)
- Maintain manual backup processes
- Extensive training program
- Vendor support and SLAs

---

## Questions to Ask

If you need more context:
1. What's driving the automation interest? (labor, capacity, accuracy, cost)
2. What's the order volume and profile? (current and projected)
3. What's the current labor situation? (cost, availability, turnover)
4. What's the capital budget and ROI expectations?
5. What's the facility situation? (size, height, constraints)
6. What technology is currently in place? (WMS, conveyors, equipment)
7. What's the timeline for implementation?
8. Are there any specific pain points or bottlenecks to address?

---

## Related Skills

- **warehouse-design**: Design warehouse layout for automation
- **order-fulfillment**: Fulfillment operations and processes
- **picker-routing-optimization**: Optimize pick paths (pre-automation)
- **warehouse-slotting-optimization**: Product placement optimization
- **process-optimization**: Improve processes before automating
- **supply-chain-analytics**: ROI analysis and performance metrics
- **computer-vision-warehouse**: Vision-based automation systems

---
name: warehouse-design
description: When the user wants to design a warehouse, optimize warehouse layout, determine facility size, or configure storage systems. Also use when the user mentions "warehouse layout," "facility design," "warehouse sizing," "storage systems," "material flow," "pick path design," "dock configuration," or "space utilization." For warehouse location selection, see facility-location-problem. For slotting existing warehouses, see warehouse-slotting-optimization.
---

# Warehouse Design

You are an expert in warehouse design and facility planning. Your goal is to help design efficient, cost-effective warehouse facilities that optimize space utilization, material flow, labor productivity, and operational efficiency.

## Initial Assessment

Before designing a warehouse, understand:

1. **Business Requirements**
   - What products will be stored? (SKU count, variety)
   - What's the throughput? (units/day, orders/day)
   - Growth projections? (3-5 year horizon)
   - Special requirements? (temperature control, hazmat, security)

2. **Operational Profile**
   - Order profile? (B2B pallets, B2C eaches, mixed)
   - Peak vs. average volume? (seasonality factor)
   - SKU velocity distribution? (fast/medium/slow movers)
   - Value-added services? (kitting, labeling, returns)

3. **Physical Constraints**
   - Site available? (dimensions, shape, constraints)
   - Building type? (new construction, existing retrofit)
   - Clear height available?
   - Column spacing and floor loading capacity?

4. **Current State**
   - Existing operations to replicate or improve?
   - Current space utilization and pain points?
   - Technology already invested in?
   - Workforce considerations?

---

## Warehouse Design Framework

### Design Principles

**1. Minimize Material Handling**
- Straight-line flow preferred
- Minimize touches and moves
- Direct putaway when possible
- Cross-docking opportunities

**2. Maximize Space Utilization**
- Vertical storage (use height)
- Dense storage for slow movers
- Efficient aisle configuration
- Right-size equipment for space

**3. Optimize Labor Productivity**
- Minimize travel distance
- Batch similar activities
- Ergonomic workstation design
- Balance workload

**4. Enable Flexibility**
- Accommodate growth
- Support multiple order types
- Scalable systems
- Adaptable layout

**5. Ensure Safety & Compliance**
- Fire codes and sprinkler requirements
- ADA accessibility
- OSHA regulations
- Product-specific regulations (food, pharma)

---

## Warehouse Sizing & Capacity

### Space Calculation Methodology

**Storage Space Required:**

```python
import numpy as np
import pandas as pd

def calculate_storage_space(sku_data, peak_factor=1.5, utilization_target=0.85):
    """
    Calculate required warehouse storage space

    Parameters:
    - sku_data: DataFrame with columns ['sku', 'avg_inventory', 'pallet_positions']
    - peak_factor: Peak inventory as multiple of average (e.g., 1.5 = 50% above avg)
    - utilization_target: Target space utilization (0.85 = 85%)

    Returns:
    - Required pallet positions
    """

    # Calculate peak inventory
    sku_data = sku_data.copy()
    sku_data['peak_inventory'] = sku_data['avg_inventory'] * peak_factor

    # Total pallet positions needed
    total_positions = sku_data['pallet_positions'].sum()

    # Adjust for utilization target (need more positions than peak to maintain flow)
    required_positions = total_positions / utilization_target

    return {
        'avg_pallet_positions': sku_data['pallet_positions'].sum(),
        'peak_pallet_positions': total_positions,
        'required_positions': round(required_positions, 0),
        'utilization_target': utilization_target
    }

# Example
sku_data = pd.DataFrame({
    'sku': [f'SKU_{i}' for i in range(1, 101)],
    'avg_inventory': np.random.randint(10, 500, 100),
    'pallet_positions': np.random.randint(1, 50, 100)
})

storage_req = calculate_storage_space(sku_data, peak_factor=1.5, utilization_target=0.85)
print(f"Required pallet positions: {storage_req['required_positions']}")
```

**Total Warehouse Square Footage:**

```python
def warehouse_sizing(storage_positions, storage_type='selective',
                     receiving_docks=10, shipping_docks=15,
                     value_added_sq_ft=5000):
    """
    Calculate total warehouse square footage

    storage_type: 'selective', 'drive-in', 'push-back', 'pallet-flow'
    """

    # Square feet per pallet position by storage type
    sq_ft_per_position = {
        'selective': 30,      # Single-deep racking, most accessible
        'double-deep': 22,    # Double-deep, less accessible
        'drive-in': 18,       # High density, LIFO
        'push-back': 20,      # High density, LIFO
        'pallet-flow': 25,    # FIFO, dynamic
        'floor-stacked': 15   # Very dense, limited access
    }

    storage_sq_ft = storage_positions * sq_ft_per_position.get(storage_type, 30)

    # Receiving area (assume 2000 sq ft per dock door)
    receiving_sq_ft = receiving_docks * 2000

    # Shipping area (assume 1500 sq ft per dock door)
    shipping_sq_ft = shipping_docks * 1500

    # Aisles and circulation (20-30% of storage)
    circulation_sq_ft = storage_sq_ft * 0.25

    # Office, break rooms, restrooms (5-10% of total)
    support_sq_ft = (storage_sq_ft + receiving_sq_ft + shipping_sq_ft) * 0.08

    # Total
    total_sq_ft = (storage_sq_ft +
                   receiving_sq_ft +
                   shipping_sq_ft +
                   circulation_sq_ft +
                   value_added_sq_ft +
                   support_sq_ft)

    breakdown = {
        'Storage': round(storage_sq_ft, 0),
        'Receiving': round(receiving_sq_ft, 0),
        'Shipping': round(shipping_sq_ft, 0),
        'Circulation': round(circulation_sq_ft, 0),
        'Value_Added': value_added_sq_ft,
        'Support': round(support_sq_ft, 0),
        'Total': round(total_sq_ft, 0)
    }

    return breakdown

# Example
sizing = warehouse_sizing(
    storage_positions=5000,
    storage_type='selective',
    receiving_docks=10,
    shipping_docks=15,
    value_added_sq_ft=5000
)

print("Warehouse Space Breakdown:")
for area, sq_ft in sizing.items():
    print(f"  {area}: {sq_ft:,.0f} sq ft")

print(f"\nTotal warehouse size: {sizing['Total']:,.0f} sq ft")
```

### Throughput Capacity Analysis

```python
def throughput_capacity(storage_positions, picks_per_day, orders_per_day,
                       lines_per_order=5, pick_rate_per_hour=100):
    """
    Analyze warehouse throughput capacity

    Parameters:
    - storage_positions: Total pallet positions
    - picks_per_day: Daily pick volume (lines)
    - orders_per_day: Daily order volume
    - lines_per_order: Average lines per order
    - pick_rate_per_hour: Picks per person per hour

    Returns:
    - Capacity analysis and labor requirements
    """

    # Labor calculations
    picks_per_day = orders_per_day * lines_per_order
    hours_per_day = 16  # Assume 2-shift operation
    pickers_needed = picks_per_day / (pick_rate_per_hour * hours_per_day)

    # Receiving capacity (pallets per dock per day)
    pallets_per_dock_day = 80  # Industry average
    receiving_capacity = pallets_per_dock_day  # per dock

    # Shipping capacity (orders per dock per day)
    orders_per_dock_day = 100  # Depends on order size
    shipping_capacity = orders_per_dock_day  # per dock

    return {
        'picks_per_day': picks_per_day,
        'pickers_needed': round(pickers_needed, 1),
        'receiving_pallets_per_dock': pallets_per_dock_day,
        'shipping_orders_per_dock': orders_per_dock_day
    }

# Example
capacity = throughput_capacity(
    storage_positions=5000,
    picks_per_day=0,  # Will calculate
    orders_per_day=2000,
    lines_per_order=5,
    pick_rate_per_hour=100
)

print(f"Daily picks: {capacity['picks_per_day']}")
print(f"Pickers needed: {capacity['pickers_needed']}")
```

---

## Warehouse Layout Design

### Layout Types

**1. U-Shaped Flow**
- Receiving and shipping on same side
- Compact footprint
- Good for smaller facilities
- Easy cross-docking

**2. Straight-Through (I-Flow)**
- Receiving on one end, shipping on opposite
- Long, narrow buildings
- Minimizes backtracking
- Clear separation of inbound/outbound

**3. L-Shaped Flow**
- Receiving on one side, shipping on adjacent
- Good for corner lots
- Moderate flow efficiency

**4. T-Shaped Flow**
- Receiving on one side, shipping splits to two sides
- Accommodates multiple shipping areas
- Complex flow patterns

### Functional Zones

```python
def design_functional_zones(total_sq_ft, order_profile='mixed'):
    """
    Allocate space to functional zones

    order_profile: 'pallet', 'case', 'each', 'mixed'
    """

    # Space allocation percentages by order profile
    allocations = {
        'pallet': {
            'Reserve_Storage': 0.50,
            'Forward_Pick': 0.10,
            'Receiving': 0.12,
            'Shipping': 0.15,
            'Value_Added': 0.03,
            'Support': 0.10
        },
        'each': {
            'Reserve_Storage': 0.35,
            'Forward_Pick': 0.25,
            'Receiving': 0.10,
            'Shipping': 0.15,
            'Value_Added': 0.05,
            'Support': 0.10
        },
        'mixed': {
            'Reserve_Storage': 0.40,
            'Forward_Pick': 0.20,
            'Receiving': 0.12,
            'Shipping': 0.15,
            'Value_Added': 0.05,
            'Support': 0.08
        }
    }

    profile_alloc = allocations.get(order_profile, allocations['mixed'])

    zones = {
        zone: round(total_sq_ft * pct, 0)
        for zone, pct in profile_alloc.items()
    }

    return zones

# Example
zones = design_functional_zones(200000, order_profile='mixed')
print("Functional Zone Allocation:")
for zone, sq_ft in zones.items():
    print(f"  {zone}: {sq_ft:,.0f} sq ft ({sq_ft/sum(zones.values())*100:.1f}%)")
```

---

## Storage Systems Selection

### Storage System Comparison

| System | Density | Selectivity | FIFO/LIFO | Cost | Best For |
|--------|---------|-------------|-----------|------|----------|
| Selective Rack | Low | 100% | Either | $ | Fast movers, high SKU count |
| Double-Deep | Medium | 50% | LIFO | $$ | Medium velocity, paired SKUs |
| Drive-In/Drive-Through | High | 10-20% | LIFO/FIFO | $$ | Slow movers, few SKUs, lots of inventory |
| Push-Back | High | 25-30% | LIFO | $$$ | High volume, limited SKUs |
| Pallet Flow | High | 100% | FIFO | $$$$ | High velocity, date-sensitive |
| Automated AS/RS | Very High | 100% | Either | $$$$$ | Very high volume, limited labor |

### Storage System Selection Logic

```python
def select_storage_system(sku_velocity, sku_diversity, fifo_required=False,
                         space_constraint=False):
    """
    Recommend storage system based on operational requirements

    Parameters:
    - sku_velocity: 'fast', 'medium', 'slow'
    - sku_diversity: 'high' (>1000 SKUs), 'medium' (100-1000), 'low' (<100)
    - fifo_required: True if FIFO needed (perishables, date-coded)
    - space_constraint: True if space is at premium
    """

    recommendations = []

    if sku_diversity == 'high':
        if sku_velocity == 'fast':
            recommendations.append({
                'system': 'Selective Racking',
                'reason': 'High SKU count requires full selectivity',
                'priority': 1
            })
            if space_constraint:
                recommendations.append({
                    'system': 'Narrow Aisle (VNA)',
                    'reason': 'Increases density for high SKU count',
                    'priority': 2
                })
        else:
            recommendations.append({
                'system': 'Selective Racking',
                'reason': 'Full selectivity for diverse SKU mix',
                'priority': 1
            })

    elif sku_diversity == 'medium':
        if sku_velocity == 'fast':
            recommendations.append({
                'system': 'Pallet Flow Rack',
                'reason': 'FIFO, high velocity, moderate SKU count',
                'priority': 1 if fifo_required else 2
            })
            recommendations.append({
                'system': 'Push-Back Rack',
                'reason': 'Dense storage, good throughput',
                'priority': 2 if fifo_required else 1
            })
        else:
            recommendations.append({
                'system': 'Double-Deep',
                'reason': 'Good density with reasonable selectivity',
                'priority': 1
            })

    else:  # low SKU diversity
        if space_constraint:
            recommendations.append({
                'system': 'Drive-In Racking',
                'reason': 'Maximum density for low SKU count',
                'priority': 1 if not fifo_required else 2
            })
            if fifo_required:
                recommendations.append({
                    'system': 'Drive-Through Racking',
                    'reason': 'Dense FIFO for low SKU count',
                    'priority': 1
                })

    # Sort by priority
    recommendations.sort(key=lambda x: x['priority'])

    return recommendations

# Example
recs = select_storage_system(
    sku_velocity='fast',
    sku_diversity='high',
    fifo_required=True,
    space_constraint=True
)

print("Storage System Recommendations:")
for rec in recs:
    print(f"  {rec['priority']}. {rec['system']}: {rec['reason']}")
```

### Rack Configuration Calculations

```python
def rack_configuration(clear_height_ft, load_height_ft=5, load_depth_ft=4,
                      aisle_width_ft=12, rack_depth='single'):
    """
    Calculate rack configuration and capacity

    Parameters:
    - clear_height_ft: Building clear height
    - load_height_ft: Height per pallet level
    - load_depth_ft: Depth per pallet position
    - aisle_width_ft: Aisle width for equipment
    - rack_depth: 'single', 'double', 'back-to-back'
    """

    # Calculate number of levels (leave clearance for sprinklers/lights)
    usable_height = clear_height_ft - 4  # 4 ft clearance
    levels = int(usable_height / load_height_ft)

    # Rack bay width (typically 96-108 inches for 2 pallets side-by-side)
    bay_width_ft = 9  # 2 x 42" pallets + structure

    # Calculate positions per bay
    if rack_depth == 'single':
        positions_per_bay = 2  # 2 pallets wide
        depth_ft = load_depth_ft + 2  # Structure
    elif rack_depth == 'double':
        positions_per_bay = 4  # 2 deep x 2 wide
        depth_ft = (load_depth_ft * 2) + 2
    elif rack_depth == 'back-to-back':
        positions_per_bay = 4  # 2 racks back to back, 2 wide each
        depth_ft = (load_depth_ft * 2) + 3  # Shared structure

    # Positions per bay
    positions_per_bay_total = positions_per_bay * levels

    # Linear feet required per bay (includes aisle)
    linear_ft_per_bay = bay_width_ft

    # Calculate density (positions per 1000 sq ft)
    sq_ft_per_bay = bay_width_ft * (depth_ft + aisle_width_ft)
    positions_per_1000_sqft = (positions_per_bay_total / sq_ft_per_bay) * 1000

    return {
        'levels': levels,
        'positions_per_bay': positions_per_bay_total,
        'bay_width_ft': bay_width_ft,
        'depth_ft': depth_ft,
        'sq_ft_per_bay': round(sq_ft_per_bay, 1),
        'positions_per_1000_sqft': round(positions_per_1000_sqft, 1)
    }

# Example
config = rack_configuration(
    clear_height_ft=32,
    load_height_ft=5,
    aisle_width_ft=12,
    rack_depth='single'
)

print(f"Rack levels: {config['levels']}")
print(f"Positions per bay: {config['positions_per_bay']}")
print(f"Density: {config['positions_per_1000_sqft']} positions per 1000 sq ft")
```

---

## Material Handling Equipment Selection

### Equipment Types & Specs

**1. Counterbalance Forklifts**
- Lift height: 15-20 ft
- Aisle width: 12-13 ft
- Use: Receiving, shipping, low-level storage
- Cost: $25K-$40K

**2. Reach Trucks**
- Lift height: 25-35 ft
- Aisle width: 8-10 ft
- Use: Selective racking, better utilization
- Cost: $35K-$50K

**3. Very Narrow Aisle (VNA) / Turret Trucks**
- Lift height: 35-45 ft
- Aisle width: 5-6.5 ft
- Use: Maximum density, high SKU count
- Cost: $60K-$100K+
- Requires wire guidance, flat floors

**4. Order Pickers**
- Lift height: 25-35 ft
- Use: Each/case picking, person-to-goods
- Cost: $40K-$70K

**5. Automated Guided Vehicles (AGV)**
- Lift height: Varies
- Use: Putaway, replenishment, goods-to-person
- Cost: $100K-$200K per vehicle

```python
def select_material_handling_equipment(storage_height_ft, aisle_budget='medium',
                                       throughput_level='medium',
                                       automation_interest=False):
    """
    Recommend material handling equipment

    Parameters:
    - storage_height_ft: How high storage needs to go
    - aisle_budget: 'tight' (<8 ft), 'medium' (8-12 ft), 'wide' (>12 ft)
    - throughput_level: 'low', 'medium', 'high'
    - automation_interest: Considering automation?
    """

    recommendations = []

    if storage_height_ft <= 15:
        recommendations.append({
            'equipment': 'Counterbalance Forklift',
            'aisle_width': '12-13 ft',
            'cost': '$25K-$40K',
            'reason': 'Low height, versatile, lowest cost'
        })

    elif storage_height_ft <= 30:
        if aisle_budget == 'tight':
            recommendations.append({
                'equipment': 'Very Narrow Aisle (VNA)',
                'aisle_width': '5-6.5 ft',
                'cost': '$60K-$100K',
                'reason': 'Maximum space utilization, tight aisles'
            })
        else:
            recommendations.append({
                'equipment': 'Reach Truck',
                'aisle_width': '8-10 ft',
                'cost': '$35K-$50K',
                'reason': 'Good balance of cost and density'
            })

    else:  # > 30 ft
        recommendations.append({
            'equipment': 'VNA or Automated AS/RS',
            'aisle_width': '5-6.5 ft or N/A',
            'cost': '$60K+ or $2M+ system',
            'reason': 'Required for very high storage'
        })

    if automation_interest and throughput_level == 'high':
        recommendations.append({
            'equipment': 'Automated Storage/Retrieval (AS/RS)',
            'aisle_width': 'N/A',
            'cost': '$2M-$10M+ system',
            'reason': 'High throughput, labor savings, accuracy'
        })

    return recommendations

# Example
equipment = select_material_handling_equipment(
    storage_height_ft=28,
    aisle_budget='medium',
    throughput_level='medium',
    automation_interest=False
)

print("Material Handling Equipment Recommendations:")
for eq in equipment:
    print(f"\n  {eq['equipment']}")
    print(f"    Aisle width: {eq['aisle_width']}")
    print(f"    Cost: {eq['cost']}")
    print(f"    Reason: {eq['reason']}")
```

---

## Dock & Receiving/Shipping Design

### Dock Door Calculations

```python
def calculate_dock_doors(inbound_trucks_per_day, outbound_trucks_per_day,
                        hours_of_operation=10, dwell_time_hours=2,
                        utilization_target=0.80):
    """
    Calculate required dock doors

    Parameters:
    - inbound_trucks_per_day: Daily truck arrivals
    - outbound_trucks_per_day: Daily truck departures
    - hours_of_operation: Working hours per day
    - dwell_time_hours: Hours per truck at dock (unload/load time)
    - utilization_target: Target dock utilization (0.80 = 80%)
    """

    # Capacity per dock door per day
    door_capacity = hours_of_operation / dwell_time_hours

    # Receiving doors needed
    receiving_doors_raw = inbound_trucks_per_day / door_capacity
    receiving_doors = receiving_doors_raw / utilization_target

    # Shipping doors needed
    shipping_doors_raw = outbound_trucks_per_day / door_capacity
    shipping_doors = shipping_doors_raw / utilization_target

    return {
        'receiving_doors': int(np.ceil(receiving_doors)),
        'shipping_doors': int(np.ceil(shipping_doors)),
        'total_doors': int(np.ceil(receiving_doors + shipping_doors)),
        'receiving_utilization': receiving_doors_raw / np.ceil(receiving_doors),
        'shipping_utilization': shipping_doors_raw / np.ceil(shipping_doors)
    }

# Example
doors = calculate_dock_doors(
    inbound_trucks_per_day=50,
    outbound_trucks_per_day=70,
    hours_of_operation=16,  # 2 shifts
    dwell_time_hours=2,
    utilization_target=0.80
)

print(f"Receiving doors needed: {doors['receiving_doors']}")
print(f"Shipping doors needed: {doors['shipping_doors']}")
print(f"Total dock doors: {doors['total_doors']}")
print(f"Receiving utilization: {doors['receiving_utilization']:.1%}")
print(f"Shipping utilization: {doors['shipping_utilization']:.1%}")
```

### Dock Configuration

**Design Considerations:**
- **Dock spacing**: 12 ft centers (standard)
- **Dock depth**: 60-80 ft for staging area
- **Turnaround space**: 130-150 ft for 53' trailers
- **Drive-in docks**: Save space but reduce throughput
- **Cross-dock**: Requires receiving and shipping proximity

---

## Pick Path & Slotting Design

### Travel Distance Calculation

```python
def calculate_pick_travel_distance(orders_per_day, lines_per_order,
                                  avg_travel_per_pick_ft=150,
                                  picker_speed_ft_per_min=200):
    """
    Calculate picker travel distance and time

    Parameters:
    - orders_per_day: Daily order volume
    - lines_per_order: Average order lines
    - avg_travel_per_pick_ft: Average travel distance per pick
    - picker_speed_ft_per_min: Walking/driving speed
    """

    picks_per_day = orders_per_day * lines_per_order

    # Total travel distance
    total_travel_ft = picks_per_day * avg_travel_per_pick_ft
    total_travel_miles = total_travel_ft / 5280

    # Travel time
    travel_time_min = total_travel_ft / picker_speed_ft_per_min
    travel_time_hours = travel_time_min / 60

    # If picks per hour = 100, calculate labor needed
    pick_time_hours = picks_per_day / 100

    total_labor_hours = pick_time_hours + travel_time_hours

    return {
        'picks_per_day': picks_per_day,
        'total_travel_miles': round(total_travel_miles, 1),
        'travel_time_hours': round(travel_time_hours, 1),
        'pick_time_hours': round(pick_time_hours, 1),
        'total_labor_hours': round(total_labor_hours, 1),
        'pickers_needed_single_shift': round(total_labor_hours / 8, 1)
    }

# Example
travel = calculate_pick_travel_distance(
    orders_per_day=2000,
    lines_per_order=5,
    avg_travel_per_pick_ft=150,
    picker_speed_ft_per_min=200
)

print(f"Daily picks: {travel['picks_per_day']}")
print(f"Daily travel: {travel['total_travel_miles']} miles")
print(f"Travel time: {travel['travel_time_hours']} hours")
print(f"Pickers needed: {travel['pickers_needed_single_shift']}")
```

### Golden Zone Placement

**Principle:**
- Place fastest movers in most accessible locations
- "Golden zone": Waist to shoulder height, front of rack
- Minimize vertical and horizontal travel

**Implementation:**
See **warehouse-slotting-optimization** skill for detailed algorithms

---

## Automation & Technology

### Warehouse Automation Options

**1. Conveyor Systems**
- Use: Move products between zones
- Cost: $200-$500 per linear foot
- ROI: 2-4 years

**2. Sortation Systems**
- Use: Sort orders/cartons to lanes/destinations
- Types: Sliding shoe, cross-belt, tilt-tray
- Cost: $500K-$3M+
- Throughput: 5K-30K+ units/hour

**3. Automated Storage & Retrieval (AS/RS)**
- Use: Dense storage, high throughput
- Types: Mini-load, unit-load, shuttle systems
- Cost: $2M-$20M+
- ROI: 3-7 years

**4. Goods-to-Person (GTP)**
- Use: Eliminate picker travel
- Types: Vertical lift modules, horizontal carousels, shuttle systems
- Cost: $500K-$5M+
- Pick rates: 200-400+ lines/person/hour

**5. Robotics**
- Types: AMRs (autonomous mobile robots), picking robots, palletizing robots
- Cost: $50K-$200K per robot
- Scalability: Add/remove as needed

**6. Automated Guided Vehicles (AGV)**
- Use: Move pallets/containers
- Cost: $100K-$200K per vehicle
- Requires infrastructure (wires/magnets or LiDAR)

```python
def automation_roi_analysis(current_labor_cost, automation_cost,
                           labor_savings_pct, maintenance_cost_annual,
                           years=5):
    """
    Calculate ROI for warehouse automation

    Parameters:
    - current_labor_cost: Annual labor cost ($)
    - automation_cost: Upfront automation investment ($)
    - labor_savings_pct: Labor reduction (e.g., 0.40 for 40%)
    - maintenance_cost_annual: Annual maintenance ($)
    - years: Analysis period
    """

    annual_savings = current_labor_cost * labor_savings_pct
    annual_net_savings = annual_savings - maintenance_cost_annual

    # Simple payback period
    payback_years = automation_cost / annual_net_savings

    # NPV calculation (assume 10% discount rate)
    discount_rate = 0.10
    npv = -automation_cost
    for year in range(1, years + 1):
        npv += annual_net_savings / ((1 + discount_rate) ** year)

    # ROI
    total_savings = annual_net_savings * years
    roi = (total_savings - automation_cost) / automation_cost

    return {
        'automation_cost': automation_cost,
        'annual_labor_savings': round(annual_savings, 0),
        'annual_maintenance': maintenance_cost_annual,
        'annual_net_savings': round(annual_net_savings, 0),
        'payback_years': round(payback_years, 2),
        'npv_5_year': round(npv, 0),
        'roi_5_year': round(roi, 2)
    }

# Example
roi = automation_roi_analysis(
    current_labor_cost=2_000_000,   # $2M annual labor
    automation_cost=4_000_000,      # $4M automation investment
    labor_savings_pct=0.50,         # 50% labor reduction
    maintenance_cost_annual=200_000, # $200K annual maintenance
    years=5
)

print(f"Automation cost: ${roi['automation_cost']:,.0f}")
print(f"Annual labor savings: ${roi['annual_labor_savings']:,.0f}")
print(f"Annual net savings: ${roi['annual_net_savings']:,.0f}")
print(f"Payback period: {roi['payback_years']} years")
print(f"5-year NPV: ${roi['npv_5_year']:,.0f}")
print(f"5-year ROI: {roi['roi_5_year']:.1%}")
```

---

## Warehouse KPIs & Performance

### Key Performance Indicators

```python
def warehouse_kpis(total_sq_ft, storage_sq_ft, occupied_positions,
                   total_positions, throughput_units, labor_hours,
                   orders_shipped, order_accuracy_pct):
    """
    Calculate warehouse performance KPIs
    """

    # Space utilization
    space_utilization = occupied_positions / total_positions

    # Inventory density
    inventory_density = occupied_positions / (storage_sq_ft / 1000)

    # Productivity
    units_per_labor_hour = throughput_units / labor_hours

    # Order accuracy
    order_accuracy = order_accuracy_pct / 100

    # Operating cost per unit (example)
    # Would need cost inputs for full calculation

    kpis = {
        'Space_Utilization_%': round(space_utilization * 100, 1),
        'Inventory_Density_per_1000sqft': round(inventory_density, 1),
        'Units_per_Labor_Hour': round(units_per_labor_hour, 1),
        'Order_Accuracy_%': round(order_accuracy * 100, 2),
        'Orders_per_Labor_Hour': round(orders_shipped / labor_hours, 1)
    }

    return kpis

# Example
kpis = warehouse_kpis(
    total_sq_ft=200000,
    storage_sq_ft=120000,
    occupied_positions=4200,
    total_positions=5000,
    throughput_units=50000,
    labor_hours=500,
    orders_shipped=2000,
    order_accuracy_pct=99.5
)

print("Warehouse KPIs:")
for metric, value in kpis.items():
    print(f"  {metric}: {value}")
```

---

## Tools & Libraries

### Design & Simulation Software

**CAD/Layout Tools:**
- **AutoCAD**: 2D/3D warehouse layout
- **SketchUp**: 3D visualization
- **Warehouse Blueprint**: Online warehouse design
- **SmartDraw**: Warehouse layout templates

**Simulation:**
- **FlexSim**: 3D simulation modeling
- **AnyLogic**: Multi-method simulation
- **Simio**: Process simulation
- **Arena**: Discrete event simulation
- **SimPy** (Python): Discrete event simulation library

**Analysis:**
- **Excel/Python**: Capacity calculations, ROI analysis
- **Tableau/Power BI**: Performance dashboards

---

## Common Challenges & Solutions

### Challenge: Insufficient Height Utilization

**Problem:**
- Low storage density
- Wasting cubic footage
- Only using 10-15 ft of 30+ ft clear height

**Solutions:**
- Install taller racking (selective or VNA)
- Use reach trucks or order pickers
- Implement mezzanines for picking
- Consider AS/RS for maximum density

### Challenge: Excessive Picker Travel

**Problem:**
- Low productivity
- High labor costs
- Pickers traveling miles per day

**Solutions:**
- Implement velocity-based slotting (see warehouse-slotting-optimization)
- Zone picking or batch picking strategies
- Goods-to-person automation
- Optimize pick path routing
- Forward pick locations for fast movers

### Challenge: Dock Congestion

**Problem:**
- Trucks waiting, detention fees
- Receiving/shipping bottleneck
- Poor appointment scheduling

**Solutions:**
- Add dock doors if capacity constrained
- Implement dock scheduling system (YMS)
- Use cross-docking to bypass storage
- Separate receiving and shipping operations
- Stagger appointments throughout day

### Challenge: Seasonal Volume Fluctuations

**Problem:**
- Facility sized for peak, underutilized off-peak
- Can't handle peak volume
- Temporary labor challenges

**Solutions:**
- Flexible storage (collapsible racks, temporary)
- Overflow space with 3PL partners
- Scale automation (add/remove AMRs)
- Cross-training workforce
- Build for average + use overflow strategy

### Challenge: Mixed Order Profiles

**Problem:**
- Some orders are pallets, some are eaches
- Difficult to optimize for both
- Separate processes inefficient

**Solutions:**
- Zone warehouse by order type
- Separate pallet and each picking areas
- Use multi-modal automation
- Different pick strategies by order type
- Consider separate facilities for very different profiles

---

## Output Format

### Warehouse Design Report

**Executive Summary:**
- Warehouse size and configuration
- Storage capacity (pallet positions)
- Throughput capacity (units/day, orders/day)
- Capital investment required
- Operating cost estimates

**Facility Specifications:**

| Specification | Value |
|---------------|-------|
| Total Square Footage | 250,000 sq ft |
| Clear Height | 32 ft |
| Storage Square Footage | 150,000 sq ft |
| Pallet Positions | 6,500 |
| Receiving Dock Doors | 12 |
| Shipping Dock Doors | 18 |
| Storage System | Selective Racking (80%), Push-Back (20%) |
| Material Handling | Reach Trucks (6), Counterbalance (4) |

**Layout Diagram:**
- Floor plan with functional zones
- Material flow diagram
- Racking layout and aisle configuration

**Capacity Analysis:**

| Metric | Design Capacity | Peak Capacity (1.2x) |
|--------|----------------|----------------------|
| Storage Positions | 6,500 | 7,800 (w/ floor stack) |
| Daily Throughput | 100,000 units | 120,000 units |
| Orders per Day | 3,000 | 3,600 |
| Receiving (pallets/day) | 800 | 960 |
| Shipping (trucks/day) | 90 | 108 |

**Investment Summary:**

| Category | Cost |
|----------|------|
| Building (if new) | $12M |
| Racking Systems | $2.5M |
| Material Handling Equipment | $800K |
| Conveyor/Sortation | $1.5M |
| IT Systems (WMS, etc.) | $500K |
| **Total Capital** | **$17.3M** |

**Operating Costs (Annual):**

| Category | Cost |
|----------|------|
| Labor | $4.5M |
| Lease/Depreciation | $2.0M |
| Utilities | $400K |
| Maintenance | $300K |
| **Total Annual** | **$7.2M** |

**Cost per Unit:** $0.72 per unit shipped

---

## Questions to Ask

If you need more context:
1. What's the throughput requirement? (units/day, orders/day, order profile)
2. What's the SKU count and velocity distribution?
3. What's the growth projection over 3-5 years?
4. Is a site identified? What are the constraints? (size, shape, clear height)
5. What's the order profile? (pallets, cases, eaches, mixed)
6. Are there special requirements? (temp control, hazmat, security)
7. What's the automation appetite and budget?
8. Greenfield or brownfield (existing building)?

---

## Related Skills

- **warehouse-slotting-optimization**: Optimize product placement within warehouse
- **warehouse-automation**: Deep dive into automation systems
- **order-fulfillment**: Pick, pack, ship operations design
- **facility-location-problem**: Where to locate the warehouse
- **picker-routing-optimization**: Optimize pick paths
- **dock-door-assignment**: Optimize dock scheduling
- **cross-docking**: Flow-through operations design
- **warehouse-location-optimization**: Network-level warehouse placement

---
name: vehicle-routing-problem
description: When the user wants to solve the Vehicle Routing Problem (VRP), optimize multi-vehicle routes, or plan fleet delivery routes. Also use when the user mentions "VRP," "fleet routing," "multi-vehicle routing," "delivery route planning," "vehicle dispatch," "fleet optimization," or "route assignment." For single vehicle, see traveling-salesman-problem. For time windows, see vrp-time-windows.
---

# Vehicle Routing Problem (VRP)

You are an expert in the Vehicle Routing Problem and fleet optimization. Your goal is to help determine optimal routes for a fleet of vehicles to serve a set of customers, minimizing total distance/cost while respecting vehicle capacities and other constraints.

## Initial Assessment

Before solving VRP instances, understand:

1. **Fleet Characteristics**
   - How many vehicles available?
   - Vehicle capacities (weight, volume, pallets)?
   - Homogeneous fleet (all same) or heterogeneous?
   - Fixed costs per vehicle vs. variable costs?
   - Maximum route duration or distance?

2. **Customer Requirements**
   - How many customers to serve?
   - Customer demands (quantities)?
   - Service time at each location?
   - Any delivery time windows? → see **vrp-time-windows**
   - Pickup and delivery? → see **pickup-delivery-problem**

3. **Problem Scale**
   - Small (< 50 customers, < 5 vehicles): Exact methods possible
   - Medium (50-200 customers): Advanced heuristics
   - Large (200+ customers): Metaheuristics, decomposition

4. **Constraints**
   - Capacity constraints?
   - Maximum route length/duration?
   - Driver breaks required?
   - Depot open hours?
   - Multiple depots? → see **multi-depot-vrp**

5. **Objectives**
   - Minimize total distance?
   - Minimize number of vehicles?
   - Minimize total cost?
   - Balance routes?

---

## Mathematical Formulation

### Capacitated VRP (CVRP) - Two-Index Formulation

**Sets:**
- V = {0, 1, ..., n}: Set of nodes (0 = depot, 1..n = customers)
- K = {1, ..., m}: Set of vehicles

**Parameters:**
- c_{ij}: Cost/distance from node i to node j
- d_i: Demand at customer i
- Q_k: Capacity of vehicle k
- M: Number of available vehicles

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j, 0 otherwise

**Objective Function:**
```
Minimize: Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each customer visited exactly once:
   Σ_{k∈K} Σ_{i∈V} x_{ijk} = 1,  ∀j ∈ V\{0}

2. Flow conservation (what goes in must come out):
   Σ_{i∈V} x_{ihk} - Σ_{j∈V} x_{hjk} = 0,  ∀h ∈ V, ∀k ∈ K

3. Vehicle starts from depot:
   Σ_{j∈V\{0}} x_{0jk} = 1,  ∀k ∈ K

4. Vehicle returns to depot:
   Σ_{i∈V\{0}} x_{i0k} = 1,  ∀k ∈ K

5. Capacity constraint:
   Σ_{i∈V\{0}} Σ_{j∈V} d_i * x_{ijk} ≤ Q_k,  ∀k ∈ K

6. Subtour elimination (various formulations):
   - MTZ constraints
   - Flow-based constraints
   - Cutset constraints

7. Binary variables:
   x_{ijk} ∈ {0,1},  ∀i,j ∈ V, ∀k ∈ K
```

### Vehicle Minimization Formulation

When minimizing number of vehicles is primary objective:

```
Minimize: Σ_{k∈K} Σ_{j∈V\{0}} x_{0jk} + α * Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

Where α is a small weight on total distance (secondary objective).

---

## Exact Algorithms

### 1. Branch-and-Cut with Set Partitioning

```python
from pulp import *
import numpy as np

def vrp_branch_and_cut_simple(dist_matrix, demands, vehicle_capacity,
                              num_vehicles, depot=0):
    """
    VRP using Branch-and-Cut (simplified version)

    Suitable for small-medium instances (up to 50 customers)

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands for each customer
        vehicle_capacity: capacity of each vehicle
        num_vehicles: number of available vehicles
        depot: depot node index

    Returns:
        dict with solution details
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Create problem
    prob = LpProblem("CVRP", LpMinimize)

    # Decision variables: x[i,j,k] = 1 if vehicle k goes from i to j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    x[i,j,k] = LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    # Objective: Minimize total distance
    prob += lpSum([dist_matrix[i][j] * x[i,j,k]
                   for i in range(n) for j in range(n) if i != j
                   for k in range(num_vehicles)]), "Total_Distance"

    # Constraints

    # 1. Each customer visited exactly once
    for j in customers:
        prob += lpSum([x[i,j,k] for i in range(n) if i != j
                      for k in range(num_vehicles)]) == 1, \
                f"Visit_{j}"

    # 2. Flow conservation
    for h in range(n):
        for k in range(num_vehicles):
            prob += (lpSum([x[i,h,k] for i in range(n) if i != h]) ==
                    lpSum([x[h,j,k] for j in range(n) if j != h])), \
                    f"Flow_{h}_{k}"

    # 3. Each vehicle leaves depot at most once
    for k in range(num_vehicles):
        prob += lpSum([x[depot,j,k] for j in customers]) <= 1, \
                f"Leave_Depot_{k}"

    # 4. Each vehicle returns to depot at most once
    for k in range(num_vehicles):
        prob += lpSum([x[i,depot,k] for i in customers]) <= 1, \
                f"Return_Depot_{k}"

    # 5. Capacity constraints (using flow-based formulation)
    for k in range(num_vehicles):
        prob += lpSum([demands[j] * lpSum([x[i,j,k] for i in range(n) if i != j])
                      for j in customers]) <= vehicle_capacity, \
                f"Capacity_{k}"

    # 6. Subtour elimination (MTZ-style)
    u = {}
    for i in customers:
        for k in range(num_vehicles):
            u[i,k] = LpVariable(f"u_{i}_{k}", lowBound=0,
                               upBound=vehicle_capacity, cat='Continuous')

    for k in range(num_vehicles):
        for i in customers:
            for j in customers:
                if i != j:
                    prob += (u[i,k] - u[j,k] + vehicle_capacity * x[i,j,k] <=
                            vehicle_capacity - demands[j]), \
                            f"Subtour_{i}_{j}_{k}"

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] == 'Optimal' or LpStatus[prob.status] == 'Feasible':
        routes = [[] for _ in range(num_vehicles)]

        for k in range(num_vehicles):
            # Build route for vehicle k
            current = depot
            route = [depot]

            while True:
                next_node = None
                for j in range(n):
                    if j != current and (current,j,k) in x:
                        if x[current,j,k].varValue > 0.5:
                            next_node = j
                            break

                if next_node is None or next_node == depot:
                    route.append(depot)
                    break

                route.append(next_node)
                current = next_node

            if len(route) > 2:  # Route has customers
                routes[k] = route

        # Remove empty routes
        routes = [r for r in routes if len(r) > 2]

        return {
            'status': LpStatus[prob.status],
            'total_distance': value(prob.objective),
            'routes': routes,
            'num_vehicles_used': len(routes),
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'total_distance': None,
            'routes': None,
            'solve_time': solve_time
        }


# Example usage
if __name__ == "__main__":
    # Example: 10 customers + 1 depot
    np.random.seed(42)

    # Generate random coordinates
    coords = np.random.rand(11, 2) * 100

    # Calculate distance matrix
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Customer demands (depot has 0 demand)
    demands = [0] + [np.random.randint(5, 20) for _ in range(10)]

    vehicle_capacity = 50
    num_vehicles = 3

    result = vrp_branch_and_cut_simple(dist_matrix, demands,
                                      vehicle_capacity, num_vehicles)

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Number of Vehicles Used: {result['num_vehicles_used']}")
    print("\nRoutes:")
    for i, route in enumerate(result['routes']):
        route_demand = sum(demands[j] for j in route[1:-1])
        print(f"  Vehicle {i+1}: {route}")
        print(f"    Demand: {route_demand}/{vehicle_capacity}")
```

---

## Classical Heuristics

### 1. Clarke-Wright Savings Algorithm

```python
def clarke_wright_savings(dist_matrix, demands, vehicle_capacity, depot=0):
    """
    Clarke-Wright Savings Algorithm for VRP

    One of the most famous VRP heuristics

    Time complexity: O(n^2 log n)
    Quality: Good solutions, fast computation

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot: depot index

    Returns:
        dict with routes and total distance
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Calculate savings s_{ij} = d_{0i} + d_{0j} - d_{ij}
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                saving = (dist_matrix[depot][i] +
                         dist_matrix[depot][j] -
                         dist_matrix[i][j])
                savings.append((saving, i, j))

    # Sort savings in descending order
    savings.sort(reverse=True)

    # Initialize: each customer in separate route
    routes = [[depot, customer, depot] for customer in customers]
    route_demands = [demands[customer] for customer in customers]

    # Merge routes based on savings
    for saving, i, j in savings:
        # Find routes containing i and j
        route_i = None
        route_j = None

        for idx, route in enumerate(routes):
            if i in route:
                route_i = idx
            if j in route:
                route_j = idx

        if route_i is None or route_j is None:
            continue

        if route_i == route_j:
            continue  # Already in same route

        # Check if i and j are at ends of their routes
        # (can only merge if they're at route ends)
        route_i_data = routes[route_i]
        route_j_data = routes[route_j]

        i_at_end = (route_i_data[1] == i or route_i_data[-2] == i)
        j_at_end = (route_j_data[1] == j or route_j_data[-2] == j)

        if not (i_at_end and j_at_end):
            continue

        # Check capacity constraint
        combined_demand = route_demands[route_i] + route_demands[route_j]
        if combined_demand > vehicle_capacity:
            continue

        # Merge routes
        # Remove depots and merge
        route_i_interior = route_i_data[1:-1]
        route_j_interior = route_j_data[1:-1]

        # Determine merge order
        if route_i_interior[-1] == i and route_j_interior[0] == j:
            new_route = [depot] + route_i_interior + route_j_interior + [depot]
        elif route_i_interior[-1] == i and route_j_interior[-1] == j:
            new_route = [depot] + route_i_interior + route_j_interior[::-1] + [depot]
        elif route_i_interior[0] == i and route_j_interior[0] == j:
            new_route = [depot] + route_i_interior[::-1] + route_j_interior + [depot]
        elif route_i_interior[0] == i and route_j_interior[-1] == j:
            new_route = [depot] + route_j_interior + route_i_interior + [depot]
        else:
            continue

        # Update routes
        routes[route_i] = new_route
        route_demands[route_i] = combined_demand

        # Remove route_j
        del routes[route_j]
        del route_demands[route_j]

    # Calculate total distance
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i+1]]

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes),
        'route_demands': route_demands
    }
```

### 2. Sweep Algorithm

```python
def sweep_algorithm(coordinates, demands, vehicle_capacity, depot_idx=0):
    """
    Sweep Algorithm for VRP

    Works with polar coordinates - sweeps around depot

    Best for: Geographically clustered customers

    Args:
        coordinates: list of (x, y) tuples
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot_idx: depot index

    Returns:
        dict with routes and total distance
    """
    import math

    n = len(coordinates)
    depot = coordinates[depot_idx]

    # Calculate polar angles from depot
    angles = []
    for i, coord in enumerate(coordinates):
        if i == depot_idx:
            continue

        dx = coord[0] - depot[0]
        dy = coord[1] - depot[1]
        angle = math.atan2(dy, dx)
        angles.append((angle, i))

    # Sort customers by angle
    angles.sort()

    # Build routes by sweeping
    routes = []
    current_route = [depot_idx]
    current_load = 0

    for angle, customer in angles:
        if current_load + demands[customer] <= vehicle_capacity:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            # Start new route
            current_route.append(depot_idx)
            routes.append(current_route)

            current_route = [depot_idx, customer]
            current_load = demands[customer]

    # Add last route
    if len(current_route) > 1:
        current_route.append(depot_idx)
        routes.append(current_route)

    # Calculate distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = math.sqrt(
                (coordinates[i][0] - coordinates[j][0])**2 +
                (coordinates[i][1] - coordinates[j][1])**2
            )

    # Calculate total distance
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i+1]]

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

### 3. Sequential Insertion Heuristics

```python
def sequential_insertion_vrp(dist_matrix, demands, vehicle_capacity,
                            depot=0, insertion_criterion='cheapest'):
    """
    Sequential insertion heuristic for VRP

    Builds routes one at a time by inserting customers

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot: depot index
        insertion_criterion: 'cheapest', 'farthest', 'nearest'

    Returns:
        dict with routes and total distance
    """
    n = len(dist_matrix)
    customers = set(range(n)) - {depot}
    routes = []

    while customers:
        # Start new route
        route = [depot]
        route_load = 0

        # Select seed customer based on criterion
        if insertion_criterion == 'farthest':
            seed = max(customers, key=lambda c: dist_matrix[depot][c])
        elif insertion_criterion == 'nearest':
            seed = min(customers, key=lambda c: dist_matrix[depot][c])
        else:  # cheapest
            seed = min(customers)

        route.append(seed)
        route.append(depot)
        route_load += demands[seed]
        customers.remove(seed)

        # Insert remaining customers into this route
        while customers:
            best_customer = None
            best_position = None
            best_cost_increase = float('inf')

            # Try inserting each customer
            for customer in customers:
                if route_load + demands[customer] > vehicle_capacity:
                    continue

                # Try each position in route
                for pos in range(1, len(route)):
                    # Cost of inserting customer at position pos
                    cost_increase = (
                        dist_matrix[route[pos-1]][customer] +
                        dist_matrix[customer][route[pos]] -
                        dist_matrix[route[pos-1]][route[pos]]
                    )

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_position = pos

            if best_customer is None:
                break  # No more customers fit in this route

            # Insert best customer
            route.insert(best_position, best_customer)
            route_load += demands[best_customer]
            customers.remove(best_customer)

        routes.append(route)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes
    )

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

---

## Improvement Heuristics

### 1. Intra-Route Improvement (2-Opt)

```python
def intra_route_2opt(route, dist_matrix):
    """
    2-opt improvement within a single route

    Args:
        route: list of node indices
        dist_matrix: distance matrix

    Returns:
        improved route
    """
    improved = True
    best_route = route.copy()

    while improved:
        improved = False
        n = len(best_route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Calculate change in distance
                delta = (
                    dist_matrix[best_route[i-1]][best_route[j]] +
                    dist_matrix[best_route[i]][best_route[j+1]] -
                    dist_matrix[best_route[i-1]][best_route[i]] -
                    dist_matrix[best_route[j]][best_route[j+1]]
                )

                if delta < -1e-10:
                    # Improvement found - reverse segment
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    improved = True
                    break

            if improved:
                break

    return best_route


def improve_all_routes_2opt(routes, dist_matrix):
    """
    Apply 2-opt to all routes

    Args:
        routes: list of routes
        dist_matrix: distance matrix

    Returns:
        improved routes and total distance
    """
    improved_routes = []

    for route in routes:
        improved_route = intra_route_2opt(route, dist_matrix)
        improved_routes.append(improved_route)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in improved_routes
    )

    return {
        'routes': improved_routes,
        'total_distance': total_distance
    }
```

### 2. Inter-Route Improvement (Cross-Exchange)

```python
def cross_exchange(routes, dist_matrix, demands, vehicle_capacity):
    """
    Cross-exchange operator between routes

    Tries swapping segments between different routes

    Args:
        routes: list of routes
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(r1 + 1, num_routes):
                route1 = routes[r1]
                route2 = routes[r2]

                # Try swapping single customers
                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Check capacity feasibility
                        customer1 = route1[i]
                        customer2 = route2[j]

                        load1 = sum(demands[c] for c in route1[1:-1])
                        load2 = sum(demands[c] for c in route2[1:-1])

                        new_load1 = load1 - demands[customer1] + demands[customer2]
                        new_load2 = load2 - demands[customer2] + demands[customer1]

                        if (new_load1 > vehicle_capacity or
                            new_load2 > vehicle_capacity):
                            continue

                        # Calculate change in distance
                        # Current edges
                        current_cost = (
                            dist_matrix[route1[i-1]][route1[i]] +
                            dist_matrix[route1[i]][route1[i+1]] +
                            dist_matrix[route2[j-1]][route2[j]] +
                            dist_matrix[route2[j]][route2[j+1]]
                        )

                        # New edges after swap
                        new_cost = (
                            dist_matrix[route1[i-1]][customer2] +
                            dist_matrix[customer2][route1[i+1]] +
                            dist_matrix[route2[j-1]][customer1] +
                            dist_matrix[customer1][route2[j+1]]
                        )

                        if new_cost < current_cost - 1e-10:
                            # Perform swap
                            route1[i] = customer2
                            route2[j] = customer1
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes


def relocate_operator(routes, dist_matrix, demands, vehicle_capacity):
    """
    Relocate operator: move customer from one route to another

    Args:
        routes: list of routes
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(num_routes):
                if r1 == r2:
                    continue

                route1 = routes[r1]
                route2 = routes[r2]

                # Try moving each customer from route1 to route2
                for i in range(1, len(route1) - 1):
                    customer = route1[i]

                    # Check if route2 can accommodate this customer
                    load2 = sum(demands[c] for c in route2[1:-1])
                    if load2 + demands[customer] > vehicle_capacity:
                        continue

                    # Try inserting at each position in route2
                    for j in range(1, len(route2)):
                        # Calculate change in distance
                        # Removal cost
                        removal_cost = (
                            dist_matrix[route1[i-1]][route1[i]] +
                            dist_matrix[route1[i]][route1[i+1]] -
                            dist_matrix[route1[i-1]][route1[i+1]]
                        )

                        # Insertion cost
                        insertion_cost = (
                            dist_matrix[route2[j-1]][customer] +
                            dist_matrix[customer][route2[j]] -
                            dist_matrix[route2[j-1]][route2[j]]
                        )

                        delta = insertion_cost - removal_cost

                        if delta < -1e-10:
                            # Perform move
                            route2.insert(j, customer)
                            route1.pop(i)
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    # Remove empty routes
    routes = [r for r in routes if len(r) > 2]

    return routes
```

---

## Metaheuristics

### 1. Genetic Algorithm for VRP

```python
import random

def genetic_algorithm_vrp(dist_matrix, demands, vehicle_capacity,
                         max_vehicles, population_size=50,
                         generations=200, mutation_rate=0.15):
    """
    Genetic Algorithm for VRP

    Args:
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity
        max_vehicles: maximum number of vehicles
        population_size: GA population size
        generations: number of generations
        mutation_rate: mutation probability

    Returns:
        best solution found
    """
    n = len(dist_matrix)
    depot = 0
    customers = list(range(1, n))

    def decode_chromosome(chromosome):
        """
        Decode chromosome into routes

        Chromosome is a permutation of customers with
        route delimiters
        """
        routes = []
        current_route = [depot]
        current_load = 0

        for gene in chromosome:
            if gene < 0:  # Route delimiter
                if len(current_route) > 1:
                    current_route.append(depot)
                    routes.append(current_route)
                current_route = [depot]
                current_load = 0
            else:  # Customer
                if current_load + demands[gene] <= vehicle_capacity:
                    current_route.append(gene)
                    current_load += demands[gene]
                else:
                    # Start new route
                    current_route.append(depot)
                    routes.append(current_route)
                    current_route = [depot, gene]
                    current_load = demands[gene]

        if len(current_route) > 1:
            current_route.append(depot)
            routes.append(current_route)

        return routes

    def calculate_fitness(chromosome):
        """Calculate fitness (inverse of total distance)"""
        routes = decode_chromosome(chromosome)

        if len(routes) > max_vehicles:
            return 0  # Infeasible

        total_distance = sum(
            sum(dist_matrix[routes[i][j]][routes[i][j+1]]
                for j in range(len(routes[i])-1))
            for i in range(len(routes))
        )

        return 1.0 / (1.0 + total_distance)

    def create_individual():
        """Create random chromosome"""
        chromosome = customers.copy()
        random.shuffle(chromosome)

        # Insert random route delimiters
        num_delimiters = random.randint(0, max_vehicles - 1)
        positions = random.sample(range(len(chromosome)), num_delimiters)

        for pos in sorted(positions, reverse=True):
            chromosome.insert(pos, -1)  # -1 is route delimiter

        return chromosome

    def crossover(parent1, parent2):
        """Order crossover"""
        # Remove delimiters
        p1_customers = [g for g in parent1 if g >= 0]
        p2_customers = [g for g in parent2 if g >= 0]

        # Perform OX crossover
        size = len(p1_customers)
        start, end = sorted(random.sample(range(size), 2))

        child = [-2] * size
        child[start:end] = p1_customers[start:end]

        pos = end
        for gene in p2_customers[end:] + p2_customers[:end]:
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1

        # Add random delimiters
        num_delimiters = random.randint(0, max_vehicles - 1)
        positions = random.sample(range(len(child)), num_delimiters)

        for pos in sorted(positions, reverse=True):
            child.insert(pos, -1)

        return child

    def mutate(chromosome):
        """Swap mutation"""
        if random.random() < mutation_rate:
            # Get customer positions (not delimiters)
            customer_positions = [i for i, g in enumerate(chromosome) if g >= 0]

            if len(customer_positions) >= 2:
                i, j = random.sample(customer_positions, 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        return chromosome

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_chromosome = None
    best_fitness = 0

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_fitness(ind) for ind in population]

        # Track best
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[fitnesses.index(max_fitness)].copy()

        # Selection and reproduction
        new_population = []

        # Elitism
        elite_count = int(0.1 * population_size)
        elite_indices = sorted(range(len(fitnesses)),
                             key=lambda i: fitnesses[i],
                             reverse=True)[:elite_count]
        new_population = [population[i].copy() for i in elite_indices]

        # Create offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = max(random.sample(list(zip(population, fitnesses)), 3),
                         key=lambda x: x[1])[0]
            parent2 = max(random.sample(list(zip(population, fitnesses)), 3),
                         key=lambda x: x[1])[0]

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    # Decode best solution
    best_routes = decode_chromosome(best_chromosome)

    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in best_routes
    )

    return {
        'routes': best_routes,
        'total_distance': total_distance,
        'num_vehicles': len(best_routes)
    }
```

### 2. Large Neighborhood Search (LNS)

```python
def large_neighborhood_search_vrp(dist_matrix, demands, vehicle_capacity,
                                 initial_routes, max_iterations=100,
                                 destroy_size=0.3):
    """
    Large Neighborhood Search for VRP

    Destroys and repairs parts of the solution iteratively

    Args:
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity
        initial_routes: initial solution
        max_iterations: number of iterations
        destroy_size: fraction of customers to remove (0-1)

    Returns:
        improved solution
    """
    import copy

    def calculate_cost(routes):
        return sum(
            sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            for route in routes
        )

    def shaw_removal(routes, num_remove):
        """
        Shaw removal: remove similar customers
        """
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])

        if not all_customers:
            return routes, []

        # Select seed customer randomly
        seed = random.choice(all_customers)

        # Calculate relatedness (based on distance)
        relatedness = []
        for customer in all_customers:
            if customer != seed:
                similarity = dist_matrix[seed][customer]
                relatedness.append((similarity, customer))

        relatedness.sort()

        # Remove most related customers
        to_remove = {seed}
        for _, customer in relatedness[:num_remove-1]:
            to_remove.add(customer)

        # Remove from routes
        new_routes = []
        for route in routes:
            new_route = [route[0]]
            for customer in route[1:-1]:
                if customer not in to_remove:
                    new_route.append(customer)
            new_route.append(route[-1])

            if len(new_route) > 2:
                new_routes.append(new_route)

        return new_routes, list(to_remove)

    def greedy_insertion(routes, removed_customers):
        """
        Greedily reinsert removed customers
        """
        depot = routes[0][0]
        uninserted = removed_customers.copy()

        while uninserted:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost_increase = float('inf')

            # Try inserting each customer
            for customer in uninserted:
                # Try each route
                for route_idx, route in enumerate(routes):
                    # Check capacity
                    current_load = sum(demands[c] for c in route[1:-1])
                    if current_load + demands[customer] > vehicle_capacity:
                        continue

                    # Try each position
                    for pos in range(1, len(route)):
                        cost_increase = (
                            dist_matrix[route[pos-1]][customer] +
                            dist_matrix[customer][route[pos]] -
                            dist_matrix[route[pos-1]][route[pos]]
                        )

                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_customer = customer
                            best_route_idx = route_idx
                            best_position = pos

            if best_customer is None:
                # Create new route
                routes.append([depot, uninserted[0], depot])
                uninserted.pop(0)
            else:
                # Insert customer
                routes[best_route_idx].insert(best_position, best_customer)
                uninserted.remove(best_customer)

        return routes

    # LNS main loop
    current_routes = copy.deepcopy(initial_routes)
    current_cost = calculate_cost(current_routes)

    best_routes = copy.deepcopy(current_routes)
    best_cost = current_cost

    all_customers = []
    for route in current_routes:
        all_customers.extend(route[1:-1])

    num_remove = max(1, int(len(all_customers) * destroy_size))

    for iteration in range(max_iterations):
        # Destroy
        partial_routes, removed = shaw_removal(current_routes, num_remove)

        # Repair
        new_routes = greedy_insertion(partial_routes, removed)

        # Evaluate
        new_cost = calculate_cost(new_routes)

        # Accept or reject (simulated annealing acceptance)
        temp = 100 * (1 - iteration / max_iterations)
        if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / temp):
            current_routes = new_routes
            current_cost = new_cost

            if new_cost < best_cost:
                best_routes = copy.deepcopy(new_routes)
                best_cost = new_cost

    return {
        'routes': best_routes,
        'total_distance': best_cost,
        'num_vehicles': len(best_routes)
    }
```

---

## Using OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_vrp_ortools(dist_matrix, demands, vehicle_capacities,
                     num_vehicles, depot=0, time_limit=30):
    """
    Solve VRP using Google OR-Tools

    Most practical and efficient approach for real-world problems

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands for each location
        vehicle_capacities: list of capacities (or single value)
        num_vehicles: number of vehicles
        depot: depot index
        time_limit: time limit in seconds

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)

    # Handle uniform fleet
    if isinstance(vehicle_capacities, (int, float)):
        vehicle_capacities = [vehicle_capacities] * num_vehicles

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 100)  # Scale for integer

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demand callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Add capacity constraints
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = True

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_distance = 0
        total_load = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_load = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                route_load += demands[node]
                index = solution.Value(routing.NextVar(index))

            route.append(manager.IndexToNode(index))

            if len(route) > 2:  # Route has customers
                routes.append(route)
                total_load += route_load

                # Calculate route distance
                route_distance = 0
                for i in range(len(route) - 1):
                    route_distance += dist_matrix[route[i]][route[i+1]]
                total_distance += route_distance

        return {
            'status': 'Optimal' if solution.ObjectiveValue() > 0 else 'Feasible',
            'routes': routes,
            'total_distance': total_distance,
            'num_vehicles_used': len(routes),
            'total_load': total_load,
            'objective_value': solution.ObjectiveValue() / 100.0
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Example usage with visualization
def visualize_vrp_solution(coordinates, routes, save_path=None):
    """
    Visualize VRP solution

    Args:
        coordinates: list of (x, y) coordinates
        routes: list of routes
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    # Plot routes
    for idx, route in enumerate(routes):
        route_coords = [coordinates[i] for i in route]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        plt.plot(xs, ys, 'o-', color=colors[idx],
                linewidth=2, markersize=8,
                label=f'Vehicle {idx+1}')

    # Plot depot
    depot = coordinates[0]
    plt.plot(depot[0], depot[1], 's', color='red',
            markersize=15, label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('VRP Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Complete example
if __name__ == "__main__":
    # Generate random problem
    np.random.seed(42)

    n = 21  # 1 depot + 20 customers
    coordinates = np.random.rand(n, 2) * 100

    # Distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Demands (depot has 0 demand)
    demands = [0] + [random.randint(5, 25) for _ in range(n-1)]

    vehicle_capacity = 100
    num_vehicles = 5

    print("Solving VRP with OR-Tools...")
    result = solve_vrp_ortools(dist_matrix, demands, vehicle_capacity,
                              num_vehicles, time_limit=30)

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Vehicles Used: {result['num_vehicles_used']}/{num_vehicles}")
    print(f"Total Load: {result['total_load']}")

    print("\nRoutes:")
    for i, route in enumerate(result['routes']):
        route_load = sum(demands[j] for j in route[1:-1])
        route_dist = sum(dist_matrix[route[j]][route[j+1]]
                        for j in range(len(route)-1))
        print(f"  Vehicle {i+1}: {route}")
        print(f"    Load: {route_load}/{vehicle_capacity}")
        print(f"    Distance: {route_dist:.2f}")

    # Visualize
    visualize_vrp_solution(coordinates, result['routes'])
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **OR-Tools (Google)**: Best for practical VRP (highly recommended)
- **PuLP**: MIP modeling
- **Pyomo**: Advanced modeling
- **VRPy**: VRP-specific library
- **python-tsp**: TSP and VRP utilities

**Metaheuristics:**
- **pymhlib**: Metaheuristic library
- **deap**: Evolutionary algorithms
- **pyswarms**: Particle swarm optimization

**Visualization:**
- **matplotlib**: Basic plots
- **plotly**: Interactive visualization
- **folium**: Map visualization with real coordinates

### Commercial Software

- **Gurobi**: State-of-art MIP solver
- **CPLEX**: IBM solver
- **Xpress**: FICO optimization
- **ORTEC**: Commercial routing software
- **Descartes**: Route planning software

### Cloud Services

- **Google Maps Platform**: Distance Matrix API
- **HERE Routing API**: Commercial routing
- **Azure Maps**: Microsoft routing service

---

## Common Challenges & Solutions

### Challenge: Problem Too Large

**Problem:**
- 500+ customers makes exact methods impractical
- Need solutions in minutes, not hours

**Solutions:**
- Use OR-Tools with time limits
- Clarke-Wright + 2-opt
- Decompose: cluster customers, solve subproblems
- Rolling horizon for dynamic problems

### Challenge: Heterogeneous Fleet

**Problem:**
- Different vehicle types (capacity, cost, speed)
- Different drivers, shifts

**Solutions:**
- Extend formulation with vehicle-specific parameters
- OR-Tools handles naturally with different capacity arrays
- Sequential approach: assign to vehicle types, then route

### Challenge: Time Windows

**Problem:**
- Customers have delivery time windows
- See **vrp-time-windows** skill

**Solutions:**
- Add temporal constraints to formulation
- Use specialized time window algorithms
- OR-Tools time window dimension

### Challenge: Real-World Distances

**Problem:**
- Euclidean distances unrealistic
- Need real road network

**Solutions:**
- Use Google Maps Distance Matrix API
- Pre-compute distance/time matrix
- Cache frequently used routes
- Consider traffic patterns

### Challenge: Dynamic Requests

**Problem:**
- New orders arrive during execution
- Need to reoptimize on the fly

**Solutions:**
- Keep routes with slack capacity
- Fast re-optimization heuristics
- Insertion methods for new customers
- Trigger re-optimization periodically

---

## Output Format

### VRP Solution Report

**Problem Instance:**
- Customers: 50
- Vehicles: 5 (capacity: 100 units each)
- Depot: Distribution Center
- Total demand: 487 units

**Solution Quality:**

| Metric | Value |
|--------|-------|
| Total Distance | 1,247.3 km |
| Vehicles Used | 5 / 5 |
| Average Route Distance | 249.5 km |
| Total Load | 487 / 500 units |
| Average Load per Vehicle | 97.4% |
| Solution Time | 8.3 seconds |

**Route Details:**

**Vehicle 1:** DC → C12 → C34 → C7 → C23 → DC
- Distance: 235.4 km
- Load: 98 / 100 units (98%)
- Customers: 4

**Vehicle 2:** DC → C5 → C18 → C42 → C31 → C9 → DC
- Distance: 278.9 km
- Load: 95 / 100 units (95%)
- Customers: 5

[Continue for all vehicles...]

**Statistics:**
- Average customers per route: 10.0
- Longest route: Vehicle 2 (278.9 km)
- Shortest route: Vehicle 5 (198.7 km)
- Total driving time: ~15.8 hours

---

## Questions to Ask

If you need more context:
1. How many customers need service? How many vehicles available?
2. What are the vehicle capacities? Homogeneous or heterogeneous fleet?
3. What units are demands measured in? (weight, volume, pallets)
4. Are there time windows for deliveries?
5. Is there a maximum route duration or distance?
6. Do you have coordinates or a distance matrix?
7. Fixed cost per vehicle or just distance-based?
8. Are there multiple depots?
9. Is this a one-time problem or recurring daily?
10. What's the acceptable solution time?

---

## Related Skills

- **traveling-salesman-problem**: For single vehicle routing
- **vrp-time-windows**: For VRP with time window constraints
- **capacitated-vrp**: For detailed CVRP formulations
- **multi-depot-vrp**: For problems with multiple depots
- **pickup-delivery-problem**: For pickup and delivery VRP
- **vrp-backhauls**: For VRP with pickups after deliveries
- **split-delivery-vrp**: For allowing multiple visits
- **route-optimization**: For practical routing applications
- **last-mile-delivery**: For final delivery mile optimization
- **fleet-management**: For overall fleet operations
