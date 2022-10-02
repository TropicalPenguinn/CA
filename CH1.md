# Computer Organization
## 개괄
* 컴퓨터에 대해서 개괄적인 소개를 다룬다.
## Classes of Computers
### PC
* A computer designed for use by an indivisual
* delivery of good performance to single users **at low cost**
* usually execute **third-party software**
> Third-party: computer program created or developed by a difference company than the one that developed the computer's operating system

### Servers
* running **larger program for multiple users** accessed only via a network
* greater computing,storage,and I/O capacity
* **dependability**

### Supercomputers
* high-end scientific and engineering caculations
* very fucking big


### Embedded computeres
* used for running one predetermined application (because so small to operate various application)
* limited function as fast as possible
* minimizing cost and power
* **lower tolerance for failure**


## Big Picture (Five Components of the computer)
![](https://velog.velcdn.com/images/everyman123/post/9942c167-8e38-4254-84aa-d14d22619b8f/image.png)

* Input/Output
* Memory
* Processor=Datapath+Control
* **Datapath** : arithmetic operations
* **Control**: tells the datapath,memory,and I/O devices what to do (OS)
![](https://velog.velcdn.com/images/everyman123/post/b0aa83fd-0430-4dd9-aec9-95cf3fe605eb/image.png)

## A Safe Place for Data 
**Memory Caching** is very important concept to Computer Architecture

### Volatile (Main) Memory
* used to hold programs while they are running (Program은 Main Memory 위에 올라가야 Process가 된다. (Process: running program))
* **Dynamic Random** Access Memory
* Dynamic: refresh data periodically to hold them
* All data in DRAM are **deleted when power if off** **(끊임없이 무언가 실행이 되어야 data가 사라지지 않는다,만약 컴퓨터를 부팅하면 RAM 위에는 아무것도 없다.)**
* It provides **random access to any location**

### Non-Volatile (Secondary) Memory
* DRAM 보다는 느리지만 싸고 대용량을 보관할 수 있다. 무엇보다 power-off에도 데이터가 남아있다. (Program이 저장되는 곳)
* Flash Memory: A nonvolatile semiconductor memory (SSD)
* Magnetic disk (hard disk)
![](https://velog.velcdn.com/images/everyman123/post/1fc3b875-3a24-4187-9552-2c78ef841cd9/image.png)

## Levels of Program Code
* 핵심: layer architecture
* 사람은 인간 친화적인 **High-level language program**으로 그로그램을 작성하면
* compiler는 그것을 **Assembly language**로 변환한다.
* 하지만 이것만으로는 메모리에 저장할 수 없기 때문에 **Assembler**가 **Binary language**로 변환한다. 
![](https://velog.velcdn.com/images/everyman123/post/e378ff7a-acd3-43d1-9fb3-90eaad5c00ca/image.png)

## Seven Great Ideas in CA
* Use Abstraction to Simplify Design: 다 알 필요는 없다. -> lower-level details are hidden to offer a simpler model at higher levels
* Make the common case fast : 자주 발생하는 일을 빠르게 만드는것이 드물게 발생하는 일을 빠르고 최적화 하는것보다 성능개선에 큰 도움을 준다. 자주 생기는 일은 단순화 하여 성능을 개선하기 쉬운 경우가 많다. 
* Performance via parallelism
* Performance via pipelining: 파이프 라이닝은 컴퓨터 구조에서 수시로 볼 수 있는 병렬성의 특별한 한 형태이다. 예를 들자면, 소방차가 없는 시대에 누군가 지른 불을 끄기 위해 사람들이 길게 늘어서서 양동이로 물을 나르는것과 비교할수있다. 이렇게 인간 사슬을 이루는 방법이 각각이 양동이를 들고 왔다갔다 하는것보다 훨씬 빠르다. 
* Performance via Prediction
* **Hierarchy of memories**
# Performance
## 개괄
**이 수업에 한해서 컴퓨터의 성능은 execution time이 얼마나 적게 걸리는지에 달렸다. execution time을 결정하는 요소는 많지만 이 수업에서는 CPU time을 줄이는 것에 집중한다.**
## Response Time as Performance
![](https://velog.velcdn.com/images/everyman123/post/cb836083-b94c-458b-ad0f-59deff392844/image.png)
* **Excution time (response time)** : The total time required for the computer **to complete a task** (CPU execution time + ....)

## Clock Cycle as Performance
* Clock cycle: single electronic pulse 주기 cycle 중에 연산과 Data transfer를 수행한다.
* Clock period: the time for a complete clock cycle
* Clock period가 짧으면 같은 시간동안 Cycle이 더 많으니 많은 연산 가능 -> Clock Rate
* Clock Rate : the inverse of the clock periode **(1초에 Clock Cycle이 얼마나 많은가 -> 연산을 1초동안 얼마나 많이 수행할 수 있는가)**
* 정리: CPU는 **Clock Cycle**(a single electronic pulse of a
CPU,)동안 연산을 수행하는데 full Clock Cycle 시간을 **Clock Period(s)**라고 한다. 만약 Clock period가 짧다면 동일 시간에 더 많은 연산이 가능할 것이다 이런 개념이 **Clock Rate(HZ)** 1초동안 Clock Cycle이 얼마나 많이 수행되는지를 나타낸다. **CPU Execution time은 Clock period에 비례하고 Clock rate에 반비례하다.**
![](https://velog.velcdn.com/images/everyman123/post/955bd840-d765-43d9-92e1-9409b3886f61/image.png)

## CPU Performance
* 간단하게 택배 안에 있는 물품을 다 꺼내고 옮긴다고 가정하자 
* 상자 갯수: Instruction count
* 상자 안 물품 평균 갯수: CPI
* 물품 하나 옮기는데 드느 시간: Clock cycle time
![](https://velog.velcdn.com/images/everyman123/post/da136c9c-3e43-43e4-acb8-62af7f6d676d/image.png)
![](https://velog.velcdn.com/images/everyman123/post/48d8bf63-2afe-4fd2-98b9-bdbb59a9ed3f/image.png)

## CPI Example (Excercise)
### 1번 문제
![](https://velog.velcdn.com/images/everyman123/post/97f944dc-7449-4566-b7d1-1a060474df59/image.png)

**박스안 물품 갯수와 하나 옮기는데 걸리는 시간 (박스 갯수는 없으니)**

$A=I*250ps*2.0=500Ips$
$B=I*500ps*1.2=600Ips$
**A is faster than B**
### 2번 문제
![](https://velog.velcdn.com/images/everyman123/post/691a9d35-2184-4340-9840-95ec5034dace/image.png)
* which code executes more instruction?
**박스 갯수**
1) $2+1+2=5$
2) $4+1+1=6$
**2 executes more instructions than 2**

* which one is faster?
**업무속도(cycle time)는 없으니 박스안에 있는 내용물 총 합을 구하자**
1) $2+2+6=10$
2) $4+2+3=9$
**2 is faster than 1**
* what is the CPI for each code
**박스안 내용물 평균 몇개가 있냐?**
1) $(2+2+6)5=2$
2) $(4+2+3)/6=1.5$



## Power Trends
![](https://velog.velcdn.com/images/everyman123/post/9c8d37d8-7ab2-41fd-a8bb-e22d379a91fb/image.png)
**clock Rate은 원하는 수준까지 왔으니 Power를 줄이자 -> 하지만 이것도 한계 -> 너무 전압이 낮아짐**

## Power Wall
**Power를 결정짓는 것이 무엇인지 살펴보고 어떻게하면 Power를 낮출 수 있는지 생각해보자**
![](https://velog.velcdn.com/images/everyman123/post/589f08d8-0043-49cc-96e0-49158ee2549d/image.png)
* **Frequency swiched: clock rate**
* **capacitive load: the number of transistors**
* 당연하게 HZ를 높이고 집적도를 높이면 전력은 많이 소모되고 또한 전압이 높을수록 전력은 많이 소모된다. 그런데 Voltage는 제곱이잖아? 그러니 **Power를 줄이는 가장 효과적인 방법은 성능을 크게 떨어트리지 않는 선에서 Voltage를 줄이는 것이다.**
![](https://velog.velcdn.com/images/everyman123/post/94812699-c29b-48ba-8faa-edf67f84b0c8/image.png)
$0.85^3=0.614125 61% relatively samll power than old CPU$
## Amdahl’s Law
모든 부품을 성능향상 시킬 수는 없다. 이때 **성능이 향상되어 줄어든 부분과 그렇지 않는 부분을 고려하여 Improvement에 따른 Execution Time을 계산하는 방식**
![](https://velog.velcdn.com/images/everyman123/post/d4186f43-2103-419c-ac5d-d8f2a09b65a1/image.png)
![](https://velog.velcdn.com/images/everyman123/post/b8e6b89a-035b-48e7-ae57-781594fe53f0/image.png)
- No Solution : 5 times faster이면 100라면 20s인데 영향 받지 않는 부분이 20s를 소요시키기 때문에 애초에 불가능함

## MIPS (Million Instructions Per Second)
(Not that good) 1초에 얼마나 많은 Instruction을 처리하나?

![](https://velog.velcdn.com/images/everyman123/post/2249dbe9-366c-4fed-bb48-7e79ff1fc6ee/image.png)

It is not good for comparing different instruction sets and their capabilities. In different architectures it takes a different quantity of instructions to accomplish the same task. -> instruction이 같은 경우만 비교 가능하고 또한 같은 program이더라도 instruction이 다르기 때문이다. **(결 국 식에서 Instruction이 빠져 생기는 문제)**

## Understanding Program Performance
* Algorithm,Language,Compiler: 상자갯수와 안에 물품을 다르게 만든다.
* Instruction set architecture (ISA) : 상자갯수 물품뿐만 아니라 옮기는 시간도 바꾼다. -> 소프트웨어에서 하드웨어로 중재자 역할을 하기 때문이다.
![](https://velog.velcdn.com/images/everyman123/post/15f0981c-dbab-4e24-b8fc-badedea627ee/image.png)

