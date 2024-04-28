from durable.lang import *

with ruleset("GNSoft"):
    @when_all(m.time == '9:00')
    def go_to_work(c):
        print("출근 시간입니다.")

    @when_all(m.time == '11:30')
    def go_to_work(c):
        print("점심 시간입니다.")

    @when_all(m.time == '18:00')
    def get_off_work(c):
        print("퇴근 시간입니다.")

    @when_all(m.business_trip == "출장 계획서")
    def trip_start(c):
        print(f"{c.m.name}가 {c.m.business_trip}를 작성했습니다.")
    
    @when_all(m.business_trip == "출장 보고서")
    def trip_end(c):
        print(f"{c.m.name}가 {c.m.business_trip}를 작성했습니다.")

    @when_all(m.worked_month >= 0)
    def vacation(c):
        print(f"{c.m.name}님의 남은 휴가 일수는 {c.m.worked_month}일 입니다.")

    @when_all(+m.my_schedule)
    def schedule_registration(c):
        print(f"{c.m.name}님의 '{c.m.my_schedule}' 일정이 등록되었습니다.")

    @when_all(+m.email_to)
    def send_email(c):
        print(f"{c.m.name}님이 {c.m.email_to}님에게 메일을 보냈습니다.")

    @when_all(c.schedule << (m.time == "10:00") & (m.day_of_week == "Monday"))
    def weeekly_conference_dept(c):
        print(f"오늘은 {c.schedule.day_of_week}이므로 {c.schedule.time}에 부서 회의가 있습니다.")

    @when_all(c.schedule << (m.time == "17:00") & (m.day_of_week == "Monday"))
    def weeekly_conference_company(c):
        print(f"오늘은 {c.schedule.day_of_week}이므로 {c.schedule.time}에 주간 보고 회의가 있습니다.")

    @when_all(c.schedule << (m.time == "17:00") & (m.day_of_week == "Friday"))
    def clean_up(c):
        print(f"오늘은 {c.schedule.day_of_week}이므로 {c.schedule.time}에 청소를 해야합니다.")

    @when_all((m.company == "지엔소프트") & (m.department == '기술연구소'))
    def introduce(c):
        c.assert_fact("GNSoft", {'company':c.m.company, 'department':c.m.department, 'name':c.m.name})

    @when_all(m.department == "기술연구소")
    def department(c):
        c.assert_fact("GNSoft", {'company':c.m.company, 'department':c.m.department, 'name':c.m.name, 'rank':'연구원'})

    @when_all(+m.name)
    def output(c):
        print(f"{c.m.company} {c.m.department} {c.m.name} {c.m.rank}")

    @when_all(m.club >= 0)
    def at_least_one(c):
        print(f"현재 {c.m.club}개의 소모임에 가입되어 있습니다.")

        if c.m.club == 0:
            print("적어도 1개의 소모임에는 가입하세요.")

post("GNSoft", {'time':"9:00", "day_of_week": "Monday"})
post("GNSoft", {'time':"10:00", "day_of_week": "Monday"})

post("GNSoft", {'time':"11:30", "day_of_week": "Tuesday"})

post("GNSoft", {'time':"17:00", "day_of_week": "Friday"})
post("GNSoft", {'time':"18:00", "day_of_week": "Friday"})

assert_fact("GNSoft", {'company':'지엔소프트', 'department':"기술연구소", 'name':'권준호'})

post("GNSoft", {'club': 0})
post("GNSoft", {'club': 3})

post("GNSoft", {'business_trip': "출장 계획서"})
post("GNSoft", {'name':'권준호', 'business_trip': "출장 계획서"})
post("GNSoft", {'name':'권준호', 'business_trip': "출장 보고서"})

post("GNSoft", {'name':'권준호', 'worked_month': 1})
post("GNSoft", {'name':'권준호', 'my_schedule': "워크숍"})
post("GNSoft", {'name':'권준호', 'my_schedule': "사업 계획서 제출"})

post("GNSoft", {'name':'권준호', 'email_to': "이대훈"})
post("GNSoft", {'name':'권준호', 'email_to': "연구소장"})