class PostProcessing(object):

    @staticmethod
    def fix_weather(domain, intent, needed_slots):
        if domain == 'آب و هوا':
            if needed_slots and intent == 'تایید' and 'COND' in needed_slots.keys() and needed_slots['COND'] == '':
                needed_slots.pop('COND')
                return domain, 'پرسش', needed_slots

        return domain, intent, needed_slots