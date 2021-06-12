/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "string.h"
#include "stdio.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define CHIP_SELECT_PORT GPIOB
#define CHIP_SELECT_PIN GPIO_PIN_6
#define RESET_PORT GPIOB
#define RESET_PIN GPIO_PIN_7
#define DATA_READY_PORT GPIOA
#define DATA_READY_PIN GPIO_PIN_15
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi2;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart1_tx;

PCD_HandleTypeDef hpcd_USB_FS;

/* USER CODE BEGIN PV */
int m_state = 0; //(0: Idle, 1: Wait for Wake, ...)
int event; //0 = event start, 1 = xbus message
uint8_t uart_buf[64];
uint8_t data[256];
struct XbusMessage{
	uint8_t m_mid;
	uint16_t m_length;
	uint8_t* m_data;
};

struct XbusMessage msg;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_I2C1_Init(void);
static void MX_SPI1_Init(void);
static void MX_SPI2_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USB_PCD_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void spi_write(uint8_t opcode, uint8_t const*data, int dataLength)
{
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_RESET);
	HAL_Delay(.04);
	uint8_t buffer[4];
	buffer[0] = opcode;
	buffer[1] = 0;
	buffer[2] = 0;
	buffer[3] = 0;
	HAL_SPI_Transmit(&hspi1, buffer, sizeof(buffer), 100);
	HAL_SPI_Transmit(&hspi1, (uint8_t*)data, dataLength, 100);
	HAL_Delay(.04);
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_SET);
}


void spi_read(uint8_t opcode, uint8_t* dest, int dataLength)
{
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_RESET);
	HAL_Delay(.04);
	uint8_t buffer[4];
	buffer[0] = opcode;
	buffer[1] = 0;
	buffer[2] = 0;
	buffer[3] = 0;
	HAL_SPI_Transmit(&hspi1, buffer, sizeof(buffer), 100);
	HAL_SPI_Receive(&hspi1, dest, dataLength, 100);
	HAL_Delay(.04);
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_SET);
}


void spi_write_raw(uint8_t const* data, int dataLength)
{
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_RESET);
	HAL_Delay(.04);
	HAL_SPI_Transmit(&hspi1, (uint8_t*)data, dataLength, 100);
	HAL_Delay(.04);
	HAL_GPIO_WritePin(CHIP_SELECT_PORT, CHIP_SELECT_PIN, GPIO_PIN_SET);
}

void print_string(char* phrase){
	strcpy((char*)uart_buf, phrase);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_value(char* phrase, int value){
	strcpy((char*)uart_buf, phrase);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	strcpy((char*)uart_buf, ": ");
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	sprintf((char*)uart_buf,"%d\r\n",value);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_data(uint8_t* xbus_data, int size){
	for(int i =0;i<size;i++){
		sprintf((char*)uart_buf,"%02X ",xbus_data[i]);
		HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	}
	strcpy((char*)uart_buf, "\r\n");
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}



void print_device_id(char* phrase, uint32_t value){
	strcpy((char*)uart_buf, phrase);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	strcpy((char*)uart_buf, ": ");
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	sprintf((char*)uart_buf,"%08lX\r\n",value);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_euler(float roll, float pitch, float yaw){
	sprintf((char*)uart_buf,"Euler: Roll = %.2f, Pitch = %.2f, Yaw = %.2f\r\n",roll,pitch,yaw);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_q(float W, float X, float Y, float Z){
	sprintf((char*)uart_buf,"Orientation: %.2f W, %.2f X, %.2f Y %.2f Z\r\n",W,X,Y,Z);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_accel(float x, float y, float z){ // Theres something wrong with the way floats are handled.... !!!
	sprintf((char*)uart_buf,"Free Accel: %.2f X, %.2f Y, %.2f Z\r\n",x,y,z);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void print_value_data(char* phrase, uint8_t* data, uint16_t length){
	uint8_t* ptr;
	ptr = data;
	strcpy((char*)uart_buf, phrase);
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	strcpy((char*)uart_buf, ": ");
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
	for(uint16_t n=0;n<length;n++){
		sprintf((char*)uart_buf,"%02X",*ptr);
		HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
		strcpy((char*)uart_buf, ", ");
		HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
		++ptr;
	}
	strcpy((char*)uart_buf, "\r\n");
	HAL_UART_Transmit(&huart1, uart_buf, strlen((char*)uart_buf), 100);
}

void reset_device(){
	HAL_GPIO_WritePin(RESET_PORT, RESET_PIN, GPIO_PIN_SET);
	HAL_Delay(1);
	HAL_GPIO_WritePin(RESET_PORT, RESET_PIN, GPIO_PIN_RESET);
}

int getMessageID(const uint8_t* xbusMessage){
	return(xbusMessage[2] & 0xff);
}

int checkDataReadyLine(){
	//print_string("in check\r\n");
	if(HAL_GPIO_ReadPin(DATA_READY_PORT, DATA_READY_PIN) == GPIO_PIN_SET){
		return 1;
	}
	return 0;

}

void readPipeStatus(uint16_t* notificationMessageSize, uint16_t* measurementMessageSize){
	uint8_t status[4];
	spi_read(0x04, status, 4);
	*notificationMessageSize = status[0] | (status[1] << 8);
	*measurementMessageSize = status[2] | (status[3] << 8);
}

void readFromPipe(uint8_t* buffer, uint16_t size, uint8_t pipe){
	spi_read(pipe,buffer,size);
}

size_t XbusMessage_createRawMessage(uint8_t* dest, struct XbusMessage* message){
	int n;
	uint8_t checksum;
	uint16_t length;
	uint8_t* dptr;
	dptr = dest;

	if(dest == 0){
		//print_string("uh oh\r\n");
		return (message->m_length < 255)? message->m_length + 7 : message->m_length + 9;
	}

	*dptr++ = 0x03; //XBUS Control Pipe
	//Fill bytes required for SPI:
	*dptr++ = 0;
	*dptr++ = 0;
	*dptr++ = 0;

	checksum = 0;
	checksum -= 0xFF; //XBUS Masterdevice

	*dptr = message->m_mid;
	checksum -= *dptr++;
	length = message->m_length;
	if(length<0xFF){ // less than XBUS Extended Length
		*dptr = length;
		checksum -= *dptr++;
	}
	else{
		*dptr = 0xFF;
		checksum -= *dptr++;
		*dptr = length >> 8;
		checksum -= *dptr++;
		*dptr = length & 0xFF;
		checksum -= *dptr++;
	}
	for(n = 0; n < message->m_length; n++){
		//print_string("ADDING DATA\r\n");
		*dptr = message->m_data[n];
		checksum-= *dptr++;
	}
	*dptr++ = checksum;
	//print_value("rawLength = ",(dptr - dest));
	return dptr - dest;
}

int checkPreamble(const uint8_t* xbusMessage){
	return (xbusMessage[0] == 0XFA);
}

uint32_t readUint32_helper(const uint8_t* data, int* index){
	uint32_t result = 0;
	result |= data[(*index)++] << 24;
	result |= data[(*index)++] << 16;
	result |= data[(*index)++] << 8;
	result |= data[(*index)++] << 0;
	return result;
}

uint16_t readUint16_helper(const uint8_t* data, int* index){
	uint16_t result = 0;
	result |= data[(*index)++] << 8;
	result |= data[(*index)++] << 0;
	return result;
}

uint8_t readUint8_helper(const uint8_t* data, int* index){
	uint8_t result = data[(*index)++];
	return result;
}

float readFloat(const uint8_t* data, int* index){
	uint32_t temp = readUint32_helper(data,index);
	float result;
	memcpy(&result, &temp, 4);
	return result;
}

void sendXbusMessage(struct XbusMessage* xbusMessage){
	uint8_t buffer[128];
	size_t rawLength = XbusMessage_createRawMessage(buffer,xbusMessage);
	//print_value_data("Xbus Message",buffer,rawLength);
	spi_write_raw(buffer,rawLength);
}



void print_sample(const uint8_t* xbus_data, int size){
	int i = 4;
	do{
		uint16_t dataID = readUint16_helper(xbus_data, &i); // <-- Grab First Data ID
		i++; // <-- account for dataSize byte
		switch(dataID){
			case(0x2030): //<-- Euler Angles
			{
				int roll = readFloat(xbus_data, &i);
				int pitch = readFloat(xbus_data, &i);
				int yaw = readFloat(xbus_data, &i);
				print_euler(roll,pitch,yaw);
			}break;
			case(0x2010): //<-- Quaternions
			{
				float W = readFloat(xbus_data, &i);
				float X = readFloat(xbus_data, &i);
				float Y = readFloat(xbus_data, &i);
				float Z = readFloat(xbus_data, &i);
				print_q(W,X,Y,Z);
			}break;
			case(0x4030):
			{
				float accelX = readFloat(xbus_data, &i);
				float accelY = readFloat(xbus_data, &i);
				float accelZ = readFloat(xbus_data, &i);
				print_accel(accelX,accelY,accelZ);
			}break;
			case(0x1020): // <-- packet counter
			{
				uint16_t packet = readUint16_helper(xbus_data, &i);
				print_value("Packet: ",packet);
				print_string("\r\n");
			}break;
			default:
			{
				print_string("Unknown Data ID :(");
			}break;
		}
	}while(i<size-1);



}

void xbusToString(const uint8_t* xbusData){ //<-- THIS IS GLOBAL DATA NOT XBUS DATA DEFINED IN OTHER FUNCTION
	if(!checkPreamble(xbusData)) print_string("Invalid Xbus Message :(\r\n");

	uint8_t messageID = getMessageID(xbusData);
	int index = 4;

	switch(messageID){
		case(62): //Wake up
		{
			print_string("XMID Wakeup Message\r\n");
		}break;
		case(1): //device ID
		{
			uint32_t deviceID = readUint32_helper(xbusData, &index);
			print_device_id("Device ID: ",deviceID);
		}break;
		case(49): //Go to Config Ack
		{
			print_string("XMID Go to Config ACK\r\n");
		}break;
		case(17): //Go to Measurement ACK
		{
			print_string("XMID Go to Measurement ACK\r\n");
		}break;
		case(54): //XMID MtData2
		{


			//Length is always data length + 4

			//print_data(xbusData, 38);





			print_sample(xbusData, 43);





		}break;
		default:
			{
				print_string("Unhandled Xbus Message... idk what that is!!\r\n");
			}
	}
}

void handleEvent(int event, const uint8_t* data){
	switch(m_state)
	{
		case(0):
		{
			if(event==0)
			{
				print_string("Resetting the device\r\n");
				reset_device();
				m_state = 1; //wait for wakeup
			}
		}break;
		case(1):
		{
			print_string("Trying to See Wakeup from Device\r\n");
			if(event == 1 && getMessageID(data) == 62) //XMID_Wakeup = 62
			{
				print_string("Got Wakeup from Device!\r\n");
				msg.m_mid = 48; //GotoConfig
				sendXbusMessage(&msg);
				m_state = 2; //wait for config
			}
		}break;
		case(2):
		{
			msg.m_mid = 0; //ReqID
			sendXbusMessage(&msg);
			m_state = 3; //wait for device ID
		}break;
		case(3):
		{
			if(event == 1 && getMessageID(data) == 1){
				print_string("Got Device ID\r\n");
				msg.m_mid = 18; // req firmware version
				sendXbusMessage(&msg);
				m_state = 4; //wait for firmware revision
			}
		}break;
		case(4):
		{
			if(event == 1 && getMessageID(data) == 19){
				print_string("Got firmware revision\r\n");
				//uint8_t xbusData[] = {0x10,0x20,0xFF,0xFF,0x20,0x30,0x00,0x0A,0x40,0x20,0x00,0x0A}; // Euler Angles AND Acceleration at 10 Hz
				//uint8_t xbusData[] = {0x10,0x20,0xFF,0xFF,0x20,0x10,0x00,0x0A,0x40,0x30,0x00,0x0A}; // <-- THIS IS PACKET COUNTER, QUATERNION w/ F32 & ENU, FREE ACCEL w/ F32 & ENU
				uint8_t xbusData[] = {0x10,0x20,0xFF,0xFF,0x40,0x30,0x00,0x0A,0x20,0x10,0x00,0x0A}; //Quaternion angles at 1 Hz
				//uint8_t xbusData[] = {0x10,0x20,0xFF,0xFF}; // Just Packet Counter
				//uint8_t xbusData[] = {0x40,0x20,0x00,0x0A}; //Acceleration at 10 Hz
				msg.m_mid = 192;  //set output config
				msg.m_length = 12; // <-- YOU HAVE TO CHANGE THIS MANUALLY
				msg.m_data = (uint8_t*)&xbusData;
				//print_value("M_MID Value is", msg.m_mid);
				//print_value("M_LENGTH Value is", msg.m_length);
				//print_value_data("M_DATA Value is", msg.m_data,msg.m_length);
				sendXbusMessage(&msg);
				//print_string("Packet Counter, Quaternion, Free Acceleration 10 Hz Config\r\n.\r\n.\r\n.\r\n");
				print_string("Just sending Acceleration");
				//print_string("Sent Acceleration at 10 Hz Config\r\n.\r\n.\r\n.\r\n");
				m_state = 5; //wait for set output configuration ack
			}
		}break;
		case(5):
		{
			if(event == 1 && getMessageID(data) == 193){
			//if(event == 1){
				print_string("Output Config Written to Device :-)\r\n");

				HAL_Delay(100);
				print_string("Sending Go to Measurement!!\r\n");
				msg.m_mid = 16;
				msg.m_length = 0;
				uint8_t empty[] = {};
				msg.m_data = empty;
				sendXbusMessage(&msg);
				m_state = 6; //READY
			}
		}break;
		case(6):
		{
			if(event == 1){
				//xbus to string (data)
				xbusToString(data);
				//print data
			}
		}break;

	}
}

void readDataFromDevice(){
	uint16_t notificationMessageSize;
	uint16_t measurementMessageSize;
	readPipeStatus(&notificationMessageSize, &measurementMessageSize);
	data[0] = 0xFA;
	data[1] = 0xFF;

	if((notificationMessageSize != 0) && (notificationMessageSize < sizeof(data))){
		readFromPipe(&data[2],notificationMessageSize, 0x05);
		handleEvent(1,data);
	}
	if((measurementMessageSize != 0) && (measurementMessageSize < sizeof(data))){
		readFromPipe(&data[2],measurementMessageSize, 0x06);
		handleEvent(1,data);
	}
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_SPI1_Init();
  MX_SPI2_Init();
  MX_USART1_UART_Init();
  MX_USB_PCD_Init();
  /* USER CODE BEGIN 2 */
  handleEvent(0,data);

  HAL_Delay(100);

  if(checkDataReadyLine()==1){
	  print_string("Data Ready Triggered");
	  readDataFromDevice();
  }
  else{
	  print_string("Did not find data ready");
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  if(checkDataReadyLine()){
		  //print_string("receiving messaage!\r\n");
		  readDataFromDevice();
	  }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE|RCC_OSCILLATORTYPE_HSI48;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLLMUL_12;
  RCC_OscInitStruct.PLL.PLLDIV = RCC_PLLDIV_3;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_I2C1
                              |RCC_PERIPHCLK_USB;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  PeriphClkInit.I2c1ClockSelection = RCC_I2C1CLKSOURCE_PCLK1;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x00000708;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi1.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_64;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief SPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI2_Init(void)
{

  /* USER CODE BEGIN SPI2_Init 0 */

  /* USER CODE END SPI2_Init 0 */

  /* USER CODE BEGIN SPI2_Init 1 */

  /* USER CODE END SPI2_Init 1 */
  /* SPI2 parameter configuration*/
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi2.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi2.Init.NSS = SPI_NSS_HARD_INPUT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 7;
  if (HAL_SPI_Init(&hspi2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI2_Init 2 */

  /* USER CODE END SPI2_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200; //9600 for science computer <---------
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USB Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_PCD_Init(void)
{

  /* USER CODE BEGIN USB_Init 0 */

  /* USER CODE END USB_Init 0 */

  /* USER CODE BEGIN USB_Init 1 */

  /* USER CODE END USB_Init 1 */
  hpcd_USB_FS.Instance = USB;
  hpcd_USB_FS.Init.dev_endpoints = 8;
  hpcd_USB_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_FS.Init.battery_charging_enable = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_Init 2 */

  /* USER CODE END USB_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel2_3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel2_3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_3_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD_R_GPIO_Port, LD_R_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6|GPIO_PIN_7, GPIO_PIN_RESET);

  /*Configure GPIO pin : MFX_IRQ_OUT_Pin */
  GPIO_InitStruct.Pin = MFX_IRQ_OUT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(MFX_IRQ_OUT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : MFX_WAKEUP_Pin PA3 PA4 PA6
                           PA7 PA15 */
  GPIO_InitStruct.Pin = MFX_WAKEUP_Pin|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_6
                          |GPIO_PIN_7|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : LD_R_Pin */
  GPIO_InitStruct.Pin = LD_R_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD_R_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PB0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : PB1 */
  GPIO_InitStruct.Pin = GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PB6 PB7 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_1_IRQn);

  HAL_NVIC_SetPriority(EXTI4_15_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI4_15_IRQn);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
